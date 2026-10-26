from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from dart_sections import HEADER_CELL_TAGS, TABLE_CELL_TAGS, _span, is_header_row
from dart_xml_sanitize import sanitize_dart_xml

DATA_DIR = Path(__file__).parent / "data"
UNIT_RE = re.compile(r"단위\s*[::]")
CAPTION_MAX_LEN = 80
# 값까지 들여다볼 속성 — 단위 출처(2)와 4단계 정형 스토어(3)를 판단하는 근거.
VALUE_ATTRS = {
    "TU": ("AUNIT", "AUNITVALUE"),
    "TE": ("ACODE", "ACONTEXT", "ADECIMAL", "ANEGATED"),
}


def _text(elem: ET.Element) -> str:
    return "".join(elem.itertext()).strip()


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def _rows(table: ET.Element) -> list[list[tuple[str, str, int, int]]]:
    rows = []
    for tr in table.iter("TR"):
        cells = [
            (c.tag, _text(c), _span(c, "colspan"), _span(c, "rowspan"))
            for c in tr
            if c.tag in TABLE_CELL_TAGS
        ]
        if cells:
            rows.append(cells)
    return rows


def _n_cols(rows: list[list[tuple[str, str, int, int]]]) -> int:
    """열 수는 colspan 합의 최대값으로 센다 — 파서(_extract_table)와 같은 기준."""
    return max(sum(colspan for _, _, colspan, _ in cells) for cells in rows)


def _leading_header_rows(rows: list[list[tuple[str, str, int, int]]], rule=is_header_row) -> int:
    """앞에서부터 헤더 행이 몇 개 연속되는지. 기본 판정은 파서의 is_header_row를 그대로 쓴다."""
    n = 0
    for cells in rows:
        if rule(cells):
            n += 1
        else:
            break
    return n


def _is_header_row_strict(cells: list[tuple[str, str, int, int]]) -> bool:
    return bool(cells) and all(tag in HEADER_CELL_TAGS for tag, _, _, _ in cells)


def _preceding_text(table: ET.Element, parents: dict[ET.Element, ET.Element]) -> str:
    parent = parents.get(table)
    if parent is None:
        return ""
    siblings = list(parent)
    for prev in reversed(siblings[: siblings.index(table)]):
        text = _text(prev)
        if text:
            return text
    return ""


def _nearest_title(elem: ET.Element, parents: dict[ET.Element, ET.Element]) -> str:
    """가장 가까운 조상 SECTION-*의 TITLE. 표지의 표들은 SECTION 바깥이라 없는 게 정상."""
    cur = parents.get(elem)
    while cur is not None:
        if cur.tag.startswith("SECTION-"):
            title = cur.find("TITLE")
            if title is not None:
                return _text(title)
        cur = parents.get(cur)
    return "(섹션 바깥 — 표지 등)"


def inspect(xml_path: Path, stats: dict, samples: list, header_gain_samples: list, want_samples: int, min_cols: int) -> None:
    root = ET.fromstring(sanitize_dart_xml(xml_path.read_text(encoding="utf-8")))
    parents = _parent_map(root)

    for table in root.iter("TABLE"):
        rows = _rows(table)
        if not rows:
            stats["빈 표(행 없음)"] += 1
            continue

        stats["표 개수"] += 1
        stats["행 수 합계"] += len(rows)
        n_cols = _n_cols(rows)
        stats["열 수 분포"][min(n_cols, 20)] += 1
        if n_cols <= 1:
            stats["1열 표(파서가 p로 되돌림)"] += 1

        for cells in rows:
            for tag, _, _, _ in cells:
                stats["셀 태그"][tag] += 1

        n_header = _leading_header_rows(rows)
        stats["선행 헤더 행 수"][n_header] += 1
        if n_cols >= min_cols:
            stats[f"선행 헤더 행 수({min_cols}열 이상)"][n_header] += 1
        if n_header == 0 and n_cols >= min_cols:
            stats["헤더 없는 다열 표"] += 1

        n_strict = _leading_header_rows(rows, _is_header_row_strict)
        if n_strict != n_header:
            stats["헤더 행 수가 구 규칙과 다른 표"] += 1
            stats["구 규칙 대비 늘어난 헤더 행 수"][n_header - n_strict] += 1
            if len(header_gain_samples) < 4 and n_cols >= min_cols:
                header_gain_samples.append(
                    {
                        "구 규칙": n_strict,
                        "새 규칙": n_header,
                        "열 수": n_cols,
                        "행": [
                            [f"{tag}[{cs}x{rs}]:{txt[:18]}" for tag, txt, cs, rs in cells]
                            for cells in rows[: n_header + 1]
                        ],
                    }
                )
        # 표 전체가 TH인 경우 — 파서는 이걸 "헤더 없음"으로 되돌린다(안 그러면 전체 행이
        # 열마다 라벨을 이어붙인 1행으로 붕괴한다).
        if n_header >= len(rows):
            stats["전체가 TH인 표(헤더 없음으로 처리)"] += 1

        # 헤더 태그가 본문 행에도 섞여 쓰이는가 — 합계 행이 첫 셀만 TH를 쓰는 케이스.
        if any(tag in HEADER_CELL_TAGS for cells in rows[n_header:] for tag, _, _, _ in cells):
            stats["본문 행에 헤더 태그 섞인 표"] += 1

        # ROWSPAN 때문에 셀 수가 열 수보다 적은 본문 행 — 격자로 펼치지 않으면 헤더 라벨과
        # 위치로 짝지을 때 열이 밀린다(회사채 표에서 2번째 행부터 5열 중 4셀).
        if any(sum(cs for _, _, cs, _ in cells) < n_cols for cells in rows[n_header:]):
            stats["ROWSPAN으로 셀이 빠진 행이 있는 표"] += 1

        cell_elems = [c for tr in table.iter("TR") for c in tr if c.tag in TABLE_CELL_TAGS]
        for elem in [table, *table.iter("TR"), *cell_elems]:
            for name in elem.attrib:
                stats[f"속성명({elem.tag})"][name] += 1
        for elem in cell_elems:
            for attr in VALUE_ATTRS.get(elem.tag, ()):
                value = elem.get(attr)
                if value is not None:
                    stats[f"{elem.tag}.{attr} 값"][value[:40]] += 1

        prev_text = _preceding_text(table, parents)
        # AUNIT은 단위 출처가 아니다. 값이 INV_GET/SH5_SEX/BASE_DT처럼 **DART 서식 필드 코드**이고
        # AUNITVALUE는 N/Y/1/20251231 같은 그 필드의 값이다 — 측정 단위가 아니다. 단위의 출처는
        # TE의 ADECIMAL(-6/-3/0/INF)이고, 스케일을 통화 단위로 옮기는 건 fact_store.py가 맡는다.
        if UNIT_RE.search(prev_text):
            stats["단위 표기 위치"]["앞 문단"] += 1
        elif any(UNIT_RE.search(t) for cells in rows[:2] for _, t, _, _ in cells):
            stats["단위 표기 위치"]["표 첫 2행"] += 1
        elif any(c.get("ADECIMAL") for c in cell_elems):
            stats["단위 표기 위치"]["TE의 ADECIMAL 속성만"] += 1
        else:
            stats["단위 표기 위치"]["없음/다른 곳"] += 1

        if prev_text and len(prev_text) <= CAPTION_MAX_LEN:
            stats["캡션 후보 있는 표"] += 1

        if len(samples) < want_samples and n_cols >= min_cols:
            samples.append(
                {
                    "위치": _nearest_title(table, parents),
                    "선행 문단": prev_text[:120],
                    "선행 헤더 행 수": n_header,
                    "열 수": n_cols,
                    "행": [
                        [f"{tag}[{cs}x{rs}]:{txt[:20]}" for tag, txt, cs, rs in cells]
                        for cells in rows[:8]
                    ],
                    "전체 행 수": len(rows),
                }
            )


XBRL_ATTRS = ("ACODE", "ACONTEXT", "ADECIMAL", "ANEGATED")
# 표시 텍스트가 숫자인지. 괄호 음수 "(1,165,837)"와 대시 "-"(값 없음)도 본다 — ANEGATED와
# 표시 부호가 어떻게 대응하는지 확인하려면 둘을 구분해서 세야 한다.
NUMERIC_RE = re.compile(r"^\(?-?[\d,]+(\.\d+)?\)?$")


def _cell_desc(cell: ET.Element) -> str:
    """셀 하나를 `태그{속성}:텍스트`로. XBRL 속성만 싣는다(HEIGHT/WIDTH 같은 서식 속성은 소음)."""
    attrs = {a: cell.get(a) for a in XBRL_ATTRS if cell.get(a) is not None}
    attr_txt = ",".join(f"{k[1:]}={v}" for k, v in attrs.items())  # ACODE -> CODE 등으로 줄여 표시
    return f"{cell.tag}{{{attr_txt}}}:{_text(cell)[:24]!r}" if attrs else f"{cell.tag}:{_text(cell)[:24]!r}"


def inspect_xbrl(xml_path: Path, stats: dict, samples: list, want_samples: int) -> None:
    """ACONTEXT가 붙은 TE 셀을 중심으로 4단계 설계에 필요한 사실을 잰다(모듈 docstring 5차)."""
    root = ET.fromstring(sanitize_dart_xml(xml_path.read_text(encoding="utf-8")))
    parents = _parent_map(root)

    for table in root.iter("TABLE"):
        trs = [tr for tr in table.iter("TR") if any(c.tag in TABLE_CELL_TAGS for c in tr)]
        ctx_cells = [c for tr in trs for c in tr if c.tag == "TE" and c.get("ACONTEXT")]
        if not ctx_cells:
            continue
        stats["ACONTEXT 셀이 있는 표"] += 1

        for cell in ctx_cells:
            ctx = cell.get("ACONTEXT")
            parts = ctx.split("_")
            stats["ACONTEXT 조각 수"][len(parts)] += 1
            stats["ACONTEXT 전체 값(상위)"][ctx] += 1
            for i, part in enumerate(parts):
                # 조각 수를 미리 알 수 없다. 키를 미리 만들어 두면 더 긴 ACONTEXT에서
                # `stats[key]`가 Counter의 기본값 0(int)을 돌려주며 죽는다.
                key = f"ACONTEXT 조각[{i:02d}]"
                if not isinstance(stats[key], Counter):
                    stats[key] = Counter()
                stats[key][part] += 1
            # 같은 셀에 ACODE가 같이 붙는가 — 붙는다면 그게 계정 식별자일 수 있다.
            stats["ACONTEXT 셀의 ACODE 유무"]["있음" if cell.get("ACODE") else "없음"] += 1
            if cell.get("ACODE"):
                stats["ACONTEXT 셀의 ACODE 값"][cell.get("ACODE")] += 1

            text = _text(cell)
            shape = (
                "빈칸" if not text
                else "대시" if text in ("-", "−", "–")
                else "괄호숫자" if text.startswith("(") and NUMERIC_RE.match(text)
                else "음수기호" if text.startswith("-") and NUMERIC_RE.match(text)
                else "숫자" if NUMERIC_RE.match(text)
                else "비숫자"
            )
            # ANEGATED=Y가 "괄호로 표시된 음수"와 겹치는지가 부호 처리 규칙을 정한다.
            stats["표시 형태 x ANEGATED"][f"{shape} / NEG={cell.get('ANEGATED', '없음')}"] += 1
            stats["표시 형태 x ADECIMAL"][f"{shape} / DEC={cell.get('ADECIMAL', '없음')}"] += 1

        # 행 단위로 본다: ACONTEXT 셀이 있는 행에서 **ACONTEXT가 없는 셀**이 라벨 후보다.
        for tr in trs:
            cells = [c for c in tr if c.tag in TABLE_CELL_TAGS]
            if not any(c.get("ACONTEXT") for c in cells):
                continue
            labels = [c for c in cells if not c.get("ACONTEXT") and _text(c)]
            stats["값 행의 라벨 셀 개수"][min(len(labels), 5)] += 1
            for c in labels:
                stats["값 행의 라벨 셀 태그/속성"][
                    c.tag + ("+ACODE" if c.get("ACODE") else "") + ("+ENG" if c.get("ENG") else "")
                ] += 1
            # 한 행 안에서 ACONTEXT 값들이 몇 종류인가 — 열마다 기간이 다르면 열=기간이다.
            stats["한 행의 ACONTEXT 종류 수"][len({c.get("ACONTEXT") for c in cells if c.get("ACONTEXT")})] += 1

        # 샘플은 **재무제표 본표**에서 뽑는다. 앞쪽 표지·회사 개요 표가 먼저 나오면 4단계 설계에
        # 필요한 본표(재무상태표·손익계산서)가 샘플에 하나도 안 들어간다.
        title = _nearest_title(table, parents)
        if len(samples) < want_samples and ("재무제표" in title or "요약재무정보" in title):
            samples.append(
                {
                    "위치": title,
                    "선행 문단": _preceding_text(table, parents)[:80],
                    "행": [
                        [_cell_desc(c) for c in tr if c.tag in TABLE_CELL_TAGS] for tr in trs[:7]
                    ],
                    "전체 행 수": len(trs),
                    # 라벨 셀의 ENG 속성 — 영문 계정명이 들어 있으면 회사 간 계정 매칭에 쓸 수 있다.
                    "라벨 ENG": [
                        c.get("ENG") for tr in trs[:7] for c in tr
                        if c.tag in TABLE_CELL_TAGS and not c.get("ACONTEXT") and c.get("ENG")
                    ][:6],
                }
            )


def main_xbrl(targets: list[dict], want_samples: int) -> None:
    stats: dict = Counter()
    names = [
        "ACONTEXT 조각 수", "ACONTEXT 전체 값(상위)", "ACONTEXT 셀의 ACODE 유무", "ACONTEXT 셀의 ACODE 값",
        "표시 형태 x ANEGATED", "표시 형태 x ADECIMAL", "값 행의 라벨 셀 개수", "값 행의 라벨 셀 태그/속성",
        "한 행의 ACONTEXT 종류 수",
    ]
    for name in names:
        stats[name] = Counter()
    samples: list = []
    for entry in targets:
        xml_path = DATA_DIR / entry["company"] / entry["period"].replace(".", "-") / f"{entry['rcept_no']}.xml"
        print(f"읽는 중: {entry['company']} {entry['period']} ({xml_path.name})")
        # 회사마다 한두 개씩 샘플이 나오게 한다 — 앞 회사가 샘플을 다 차지하면 회사 간 차이가 안 보인다.
        before = len(samples)
        per_company: list = []
        inspect_xbrl(xml_path, stats, per_company, 2)
        samples.extend(per_company[: max(0, want_samples - before)])

    print(f"\n{'=' * 60}\nXBRL 속성 진단 — 필링 {len(targets)}건\n{'=' * 60}")
    # 조각 카운터는 등장 순서대로 생기므로 번호순으로 모아 맨 뒤에 찍는다.
    pieces = sorted(k for k in stats if k.startswith("ACONTEXT 조각["))
    ordered = [k for k in stats if k not in pieces] + pieces
    for key, value in ((k, stats[k]) for k in ordered):
        if isinstance(value, Counter):
            if not value:
                continue
            print(f"\n{key}:")
            # 전체 값은 잘리지 않게 폭을 넓힌다(이 진단의 목적이 잘린 뒷부분을 보는 것이다).
            width = 70 if key == "ACONTEXT 전체 값(상위)" else 32
            for k, v in sorted(value.items(), key=lambda kv: (-kv[1], str(kv[0])))[:15]:
                print(f"  {str(k):<{width}} {v:,}")
        else:
            print(f"{key}: {value:,}")

    for i, s in enumerate(samples, 1):
        print(f"\n{'=' * 60}\nXBRL 샘플 표 {i} — {s['위치']} (전체 {s['전체 행 수']}행)\n{'=' * 60}")
        print(f"선행 문단: {s['선행 문단']!r}")
        print("(형식: 태그{XBRL속성}:텍스트 — CONTEXT/DECIMAL/NEGATED/CODE는 앞 A를 뗀 이름)")
        for cells in s["행"]:
            print("  " + " | ".join(cells))
        if s["라벨 ENG"]:
            print(f"  라벨 셀 ENG 속성: {s['라벨 ENG']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DART 표 구조 측정 진단 (읽기 전용)")
    ap.add_argument("--company", help="회사명으로 한정")
    ap.add_argument("--period", help="기간으로 한정 (예: 2025-12)")
    ap.add_argument("--samples", type=int, default=2, help="원형을 출력할 표 개수")
    ap.add_argument("--min-cols", type=int, default=4, help="샘플로 뽑을 표의 최소 열 수(레이아웃 표 제외용)")
    ap.add_argument("--xbrl", action="store_true", help="4단계 설계용 XBRL 속성 진단(ACONTEXT 전체 값·라벨 셀·부호)")
    args = ap.parse_args()

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path}가 없습니다.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    targets = [
        e for e in manifest
        if (args.company is None or e["company"] == args.company)
        and (args.period is None or e["period"].replace(".", "-") == args.period)
    ]
    if not targets:
        raise SystemExit("조건에 맞는 필링이 없습니다.")
    if args.xbrl:
        # 샘플은 회사당 최대 2개, 기본 10개(5개사 x 2) — --samples로 조절
        main_xbrl(targets, max(args.samples, 10))
        return

    stats: dict = Counter()
    counters = [
        "셀 태그", "선행 헤더 행 수", f"선행 헤더 행 수({args.min_cols}열 이상)",
        "단위 표기 위치", "열 수 분포", "구 규칙 대비 늘어난 헤더 행 수",
    ]
    counters += [f"속성명({tag})" for tag in ("TABLE", "TR", *TABLE_CELL_TAGS)]
    counters += [f"{tag}.{attr} 값" for tag, attrs in VALUE_ATTRS.items() for attr in attrs]
    for name in counters:
        stats[name] = Counter()
    samples: list = []
    header_gain_samples: list = []

    print(f"헤더로 볼 태그: {HEADER_CELL_TAGS}")
    for entry in targets:
        xml_path = DATA_DIR / entry["company"] / entry["period"].replace(".", "-") / f"{entry['rcept_no']}.xml"
        print(f"읽는 중: {entry['company']} {entry['period']} ({xml_path.name})")
        inspect(xml_path, stats, samples, header_gain_samples, args.samples, args.min_cols)

    print(f"\n{'=' * 60}\n필링 {len(targets)}건 집계\n{'=' * 60}")
    for key, value in stats.items():
        if isinstance(value, Counter):
            if not value:
                continue
            print(f"\n{key}:")
            for k, v in sorted(value.items(), key=lambda kv: (-kv[1], str(kv[0])))[:12]:
                print(f"  {str(k):<24} {v:,}")
        else:
            print(f"{key}: {value:,}")

    for i, s in enumerate(samples, 1):
        print(f"\n{'=' * 60}\n샘플 표 {i} ({s['열 수']}열) — {s['위치']}\n{'=' * 60}")
        print(f"선행 문단: {s['선행 문단']!r}")
        print(f"선행 헤더 행 수: {s['선행 헤더 행 수']} / 전체 {s['전체 행 수']}행")
        print("(형식: 태그[colspan x rowspan]:셀텍스트)")
        for cells in s["행"]:
            print("  " + " | ".join(cells))

    for i, s in enumerate(header_gain_samples, 1):
        print(f"\n{'=' * 60}\n규칙 변경으로 헤더가 늘어난 표 {i} ({s['열 수']}열)")
        print(f"구 규칙 {s['구 규칙']}행 -> 새 규칙 {s['새 규칙']}행  (마지막 줄은 첫 본문 행)\n{'=' * 60}")
        for j, cells in enumerate(s["행"]):
            mark = "H" if j < s["새 규칙"] else "B"
            print(f"  [{mark}] " + " | ".join(cells))


if __name__ == "__main__":
    main()
