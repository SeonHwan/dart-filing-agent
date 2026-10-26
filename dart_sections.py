from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

SECTION_TAGS = ("SECTION-1", "SECTION-2", "SECTION-3")
TABLE_CELL_TAGS = ("TD", "TH", "TU", "TE")
# 헤더 태그는 TH **하나뿐**이다. 15건 전량 측정으로 확정(inspect_tables.py):
#   TH  100,116개 — ACOPYCOL/ADELETECOL/AMOVECOL(열 단위 편집 속성) + COLSPAN 15,548 -> 열 헤더
#   TE  368,954개 — ACODE 169,383 / ACONTEXT / ADECIMAL / ANEGATED -> XBRL식 태깅된 수치 데이터 셀
#   TU   10,958개 — AUNIT / AUNITVALUE 전량 -> 단위·유형이 붙은 값 셀 ("주권상장법인" 등)
# TU/TE를 헤더로 넣으면 **첫 여러 행이 전부 TE인 재무제표**가 5~11행짜리 헤더를 가진 표로
# 오인된다(그런 표가 각 800개씩 있다).
HEADER_CELL_TAGS = ("TH",)
CELL_SEP = " | "
# "(단위 : 백만원)", "단위: 원" 등. 표와 떨어진 별개 문단으로 존재하는 경우가 있어서
# 표 블록에 따로 실어둔다 — 단위 없는 숫자 표는 검색돼도 답변에 쓸 수 없다.
UNIT_RE = re.compile(r"단위\s*[::]\s*[^)\]\s]+")
CAPTION_MAX_LEN = 80
# 라벨용 1행 표의 라벨 길이 상한(_label_table 참고). 관측된 라벨은 "전기"/"당기"/"금융자산의 범주별
# 공시" 등 30자 이내였다 — 더 길면 내용이 있는 1행 표로 본다.
LABEL_TABLE_MAX_LEN = 30
# 정보가 없는 탐색용 문구. DART 뷰어의 "본문으로 돌아가기" 링크 텍스트라 XII. 상세표에만 나온다
# (15건 91개, 전부 1열 표 셀 안의 `<A REFNO=...>`). 청크가 되면 4~7토큰짜리가 되어 무관한 질의의
# hybrid 상위에 올라온다(`sk-risk` dump 4·6·8·10·12·14위). 문단을 통째로 비교한다 — 부분 일치로 지우면
# 같은 문구를 인용한 본문까지 잃는다.
NAV_TEXTS = frozenset({"☞ 본문 위치로 이동"})
# COLSPAN/ROWSPAN 상한 — 격자가 span만큼 칸을 채우므로 비정상 값이 오면 폭발한다(_span 참고).
MAX_SPAN = 200


@dataclass
class Section:
    level: int
    atocid: str | None
    title_kr: str
    title_en: str | None
    path: list[str]  # 루트부터의 제목 경로(breadcrumb) — 멀티문서 라우팅의 분류 축이 될 부분
    blocks: list[dict]  # 이 섹션에 "직속"된 내용만(하위 섹션 내용은 안 섞임)
    children: list[Section] = field(default_factory=list)

    def iter_leaves(self):
        """내용을 담은 말단 섹션(자기 자신 포함, 자식 유무 무관)을 전부 순회."""
        yield self
        for child in self.children:
            yield from child.iter_leaves()


def _elem_text(elem: ET.Element) -> str:
    return "".join(elem.itertext()).strip()


def _span(cell: ET.Element, name: str) -> int:
    raw = cell.get(name.upper()) or cell.get(name.lower()) or "1"
    try:
        return min(MAX_SPAN, max(1, int(raw.strip())))
    except (ValueError, AttributeError):
        return 1


def _strip_unit(text: str) -> str:
    """캡션 후보에서 단위 표기를 떼어낸다. 단위만 적힌 줄이면 빈 문자열이 된다(캡션 아님)."""
    cleaned = UNIT_RE.sub("", text)
    # 단위를 빼고 남은 껍데기 괄호와 구분기호를 정리: "가. 매출실적 ()" -> "가. 매출실적"
    cleaned = re.sub(r"\(\s*\)|\[\s*\]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" .,:-·")


def _cell_text(cell: ET.Element) -> str:
    return re.sub(r"\s+", " ", _elem_text(cell)).strip()


def _cells(tr: ET.Element) -> list[tuple[str, str, int, int]]:
    return [
        (c.tag, _cell_text(c), _span(c, "colspan"), _span(c, "rowspan"))
        for c in tr
        if c.tag in TABLE_CELL_TAGS
    ]


def is_header_row(cells: list[tuple[str, str, int, int]]) -> bool:
    if not cells:
        return False
    tags = [tag for tag, _, _, _ in cells]
    return any(tag in HEADER_CELL_TAGS for tag in tags) and "TD" not in tags


def _leading_header_count(rows: list[list[tuple[str, str, int, int]]]) -> int:
    n = 0
    for cells in rows:
        if is_header_row(cells):
            n += 1
        else:
            break
    return n


def _grid(rows: list[list[tuple[str, str, int, int]]]) -> list[list[str]]:
    grid: dict[tuple[int, int], str] = {}
    for r, cells in enumerate(rows):
        c = 0
        for _, text, colspan, rowspan in cells:
            while (r, c) in grid:
                c += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = text
            c += colspan

    if not grid:
        return []
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def _merge_header_rows(header_grid: list[list[str]]) -> list[str]:
    """격자로 펼친 헤더 행들을 열마다 라벨 하나로 합성한다("매출액" + "제58기" -> "매출액 제58기")."""
    if not header_grid:
        return []
    labels = []
    for c in range(len(header_grid[0])):
        parts: list[str] = []
        for row in header_grid:
            text = row[c]
            # 병합으로 같은 라벨이 세로로 반복되면 한 번만 쓴다("매출액 매출액 제58기" 방지).
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        labels.append(" ".join(parts))
    return labels


def _label_table(table: ET.Element, table_follows: bool = False) -> tuple[str, str] | None:
    rows = [cells for cells in (_cells(tr) for tr in table.iter("TR")) if any(t for _, t, _, _ in cells)]
    if len(rows) != 1:
        return None
    texts = list(dict.fromkeys(t for _, t, _, _ in rows[0] if t))
    units = [t for t in texts if UNIT_RE.search(t)]
    labels = [t for t in texts if not UNIT_RE.search(t)]
    if len(labels) > 1 or (labels and len(labels[0]) > LABEL_TABLE_MAX_LEN):
        return None
    if units:
        return (labels[0] if labels else ""), UNIT_RE.search(units[0]).group(0)
    if table_follows and labels:
        return labels[0], ""
    return None


def _extract_table(table: ET.Element, caption: str, unit: str) -> dict | None:
    rows = [cells for cells in (_cells(tr) for tr in table.iter("TR")) if any(t for _, t, _, _ in cells)]
    if not rows:
        return None

    # 헤더와 본문의 열 위치를 맞추려면 격자를 표 전체에 한 번에 적용해야 한다. 헤더 셀의
    # ROWSPAN이 본문 행까지 내려오는 경우가 있어서 따로 펼치면 열이 어긋난다.
    grid = _grid(rows)
    if not grid:
        return None

    # DART는 표를 레이아웃 용도로도 쓴다 — 39,307개 중 **15,954개(41%)가 1열**이고
    # (표지의 날짜 칸, "제출대상법인 유형 : 주권상장법인" 같은 서식 블록), 열이 하나면
    # 헤더도 셀 구분자도 의미가 없다. 이런 건 표 기계를 통과시키지 않고 문단으로 되돌린다 —
    # 안 그러면 캡션 줄만 덧붙은 표 청크가 되어 검색 결과의 "표 비율"만 부풀린다.
    if len(grid[0]) <= 1:
        # 탐색용 링크 문구(NAV_TEXTS)는 전부 이 경로로 들어온다 — 1열 표의 `<A REFNO=...>` 셀.
        text = "\n".join(row[0] for row in grid if row[0] and row[0] not in NAV_TEXTS)
        return {"type": "p", "text": text} if text else None

    n_header = _leading_header_count(rows)
    # 표 전체가 TH인 경우(5~11행 헤더 꼬리에 섞여 있음)에는 헤더가 없다고 본다.
    # 헤더로 두면 본문이 비어서 "헤더를 본문으로 승격"하는 경로를 타는데, 그러면 11행이
    # 열마다 라벨을 이어붙인 1행으로 붕괴해 표 구조가 사라진다.
    if n_header >= len(grid):
        n_header = 0

    header = _merge_header_rows(grid[:n_header])
    body = [row for row in grid[n_header:] if any(row)]
    if not body:
        return None

    if not unit:
        for cells in rows[:2]:
            for _, text, _, _ in cells:
                match = UNIT_RE.search(text)
                if match:
                    unit = match.group(0)
                    break
            if unit:
                break

    return {"type": "table", "caption": caption, "unit": unit, "header": header, "rows": body}


def serialize_table(block: dict, with_header: bool = True) -> str:
    lines = []
    if with_header and block.get("header"):
        lines.append(CELL_SEP.join(block["header"]))
    lines.extend(CELL_SEP.join(row) for row in block["rows"])
    return "\n".join(lines)


def table_caption_line(block: dict) -> str:
    """표 청크의 문맥 접두사에 넣을 한 줄(캡션 + 단위)."""
    parts = [p for p in (block.get("caption"), block.get("unit")) if p]
    return "표: " + " ".join(parts) if parts else "표"


def _direct_blocks(elem: ET.Element) -> list[dict]:
    blocks: list[dict] = []
    caption, unit = "", ""
    # 라벨용 1행 표(_label_table)는 블록으로 내보내지 않고 바로 다음 표의 (캡션, 단위)로 넘긴다.
    pending: tuple[str, str] | None = None

    def flush_pending() -> None:
        # 뒤에 표가 오지 않으면(15건에서 5건 안팎) 라벨 내용을 잃지 않게 문단으로 남긴다.
        nonlocal pending
        if pending:
            blocks.append({"type": "p", "text": " ".join(p for p in pending if p)})
        pending = None

    def add_table(table: ET.Element, cap: str, un: str, table_follows: bool = False) -> None:
        nonlocal pending
        label = _label_table(table, table_follows)
        if label:
            if pending:
                # 라벨 표가 연달아 오면 합친다. 예: `전기` 표 바로 뒤에 `(단위 : 천원)` 표가 따로
                # 오는 경우(삼성바이오로직스 연결재무제표 주석) — 합치지 않으면 앞 라벨이 문단으로 떨어져
                # "전기" 한 단어짜리 청크가 된다.
                cap = " ".join(p for p in (pending[0], cap) if p)
                un = pending[1] or un
                pending = None
            # 앞 문단 캡션이 있으면 라벨을 이어 붙인다("(1) 금융자산" + "전기").
            pending = (" ".join(p for p in (cap, label[0]) if p), label[1] or un)
            return
        if pending:
            cap, un = pending
            pending = None
        block = _extract_table(table, cap, un)
        if block:
            blocks.append(block)

    for child in elem:
        if child.tag in SECTION_TAGS or child.tag in ("TITLE", "LIBRARY"):
            continue

        if child.tag == "P":
            text = _elem_text(child)
            if not text or text in NAV_TEXTS:
                continue
            flush_pending()
            blocks.append({"type": "p", "text": text})
            match = UNIT_RE.search(text)
            unit = match.group(0) if match else ""
            # 캡션에서는 단위를 떼어낸다. 안 그러면 table_caption_line이 캡션과 unit을 붙일 때
            # "표: 가. 매출실적 (단위 : 백만원) 단위 : 백만원"처럼 같은 말이 두 번 들어간다.
            stripped = _strip_unit(text) if len(text) <= CAPTION_MAX_LEN else ""
            # **단위만 적힌 문단은 앞 캡션을 지우지 않는다.** `(1) 주요 제품별 매출실적` 다음 줄이
            # `(단위 : 억원)`인 경우가 흔한데, 단위 문단이 캡션을 ""로 덮어쓰면 표가 제목을 잃고
            # 제목 문단은 20토큰 미만 독립 청크로 남는다.
            if stripped or not match:
                caption = stripped

        elif child.tag == "TABLE":
            add_table(child, caption, unit)
            caption, unit = "", ""  # 캡션은 바로 뒤 표 하나에만 쓴다

        elif child.tag == "TABLE-GROUP":
            tables = list(child.iter("TABLE"))
            for i, table in enumerate(tables):
                # 그룹 안 두 번째 표부터는 캡션을 물려주지 않는다(라벨 표가 넘긴 것은 예외).
                first = i == 0
                add_table(table, caption if first else "", unit if first else "", i + 1 < len(tables))
            caption, unit = "", ""

        else:
            # PGBRK(쪽 나눔) 등. 라벨 표와 본 표 사이에 끼는 경우가 있어 pending은 유지한다.
            caption, unit = "", ""

    flush_pending()
    return blocks


def _walk(elem: ET.Element, level: int, path: list[str]) -> Section:
    title_elem = elem.find("TITLE")
    title_kr = _elem_text(title_elem) if title_elem is not None else ""
    title_en = title_elem.get("ENG") if title_elem is not None else None
    atocid = title_elem.get("ATOCID") if title_elem is not None else None
    my_path = path + [title_kr]

    section = Section(
        level=level,
        atocid=atocid,
        title_kr=title_kr,
        title_en=title_en,
        path=my_path,
        blocks=_direct_blocks(elem),
    )

    next_tag = f"SECTION-{level + 1}"
    containers = [elem, *elem.findall("LIBRARY")]
    for container in containers:
        for child in container.findall(next_tag):
            section.children.append(_walk(child, level + 1, my_path))

    return section


def parse_sections(root: ET.Element) -> list[Section]:
    """BODY 바로 아래 SECTION-1들을 최상위 챕터 트리로 반환."""
    body = root.find("BODY")
    return [_walk(s1, 1, []) for s1 in body.findall("SECTION-1")]
