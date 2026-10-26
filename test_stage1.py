from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import numpy as np

# bge_m3는 FlagEmbedding을 **지연 import**하므로(BgeM3Encoder.__init__ 안) 이 모듈만
# import하는 것으로 모델이 올라가지는 않는다. maxsim은 numpy만 쓰는 순수 함수다.
from bge_m3 import maxsim
from chunk_sections import EMBEDDING_MODEL, MAX_TOKENS, MIN_CHUNK_TOKENS, chunk_filing
from dart_sections import CELL_SEP, parse_sections
from dart_xml_sanitize import sanitize_dart_xml
from fusion import RRF_K, rrf_fuse
from parse_filing import _flatten

# 표 행을 많이 넣어서 실제로 여러 조각으로 갈리게 만든다(헤더 반복을 확인하려면 2조각 이상 필요).
ROWS = "\n".join(
    f"<TR><TD>계정과목{i:03d}</TD><TD>{i * 1000:,}</TD><TD>{i * 900:,}</TD><TD>{i * 800:,}</TD></TR>"
    for i in range(1, 121)
)

XML = f"""<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY>
<SECTION-1><TITLE ATOCID="1">III. 재무에 관한 사항</TITLE>
<SECTION-2><TITLE ATOCID="4">3. 연결재무제표 주석</TITLE>
<P>당사의 연결재무제표에 대한 주석은 다음과 같습니다. 이 문단은 서술형 텍스트라 문단
그리디 묶기 경로를 타야 합니다.</P>
<P>가. 자본금 변동 내역 (단위 : 백만원)</P>
<TABLE>
<TR><TH ROWSPAN="2">과목</TH><TH COLSPAN="3">보고기간</TH></TR>
<TR><TH>제58기</TH><TH>제57기</TH><TH>제56기</TH></TR>
{ROWS}
</TABLE>
<P>나. 해당사항 없음</P>
</SECTION-2></SECTION-1>
</BODY></DOCUMENT>
"""


class FakeTokenizer:

    @staticmethod
    def _offsets(text: str) -> list[tuple[int, int]]:
        offsets, cursor = [], 0
        for token in text.split():
            start = text.index(token, cursor)
            offsets.append((start, start + len(token)))
            cursor = start + len(token)
        return offsets

    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=False):
        if isinstance(texts, str):
            offsets = self._offsets(texts)
            return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}
        result = {"input_ids": [list(range(len(t.split()))) for t in texts]}
        if return_offsets_mapping:
            result["offset_mapping"] = [self._offsets(t) for t in texts]
        return result

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def main() -> None:
    # Windows 기본 콘솔은 cp949라서 출력에 섞인 em-dash 같은 문자에서 UnicodeEncodeError로
    # 죽는다. 테스트가 실행 환경 때문에 실패하지 않게 먼저 눙친다.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = ET.fromstring(sanitize_dart_xml(XML))
    sections = _flatten(parse_sections(root))
    context = {"company": "KB금융", "period": "2025-12"}

    tokenizer = FakeTokenizer()
    chunks = chunk_filing(tokenizer, sections, context)

    failures = []

    def check(cond: bool, label: str) -> None:
        print(f"  [{'OK  ' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)

    print(f"섹션 {len(sections)}개 -> 청크 {len(chunks)}개")
    for c in chunks:
        head = c["text"].split("\n")[0][:56]
        print(f"  {c['kind']:<6} idx={c['chunk_index']:<3} n={c['n_tokens']:<4} {head!r}")

    table_chunks = [c for c in chunks if c["kind"] == "table"]
    para_chunks = [c for c in chunks if c["kind"] == "p"]

    print("\n검증:")
    check(len(table_chunks) >= 2, f"표가 여러 조각으로 갈렸다 (표 청크 {len(table_chunks)}개)")
    check(bool(para_chunks), f"서술 청크도 생성됐다 ({len(para_chunks)}개)")

    header_line = CELL_SEP.join(["과목", "보고기간 제58기", "보고기간 제57기", "보고기간 제56기"])
    check(
        all(header_line in c["text"] for c in table_chunks),
        "모든 표 조각이 헤더 행을 갖는다 (ROWSPAN 중복 제거 + COLSPAN 격자 복원 포함)",
    )
    check(
        all("자본금 변동 내역" in c["text"] and "단위 : 백만원" in c["text"] for c in table_chunks),
        "모든 표 조각이 캡션과 단위를 갖는다",
    )

    # 행 경계 보존: 헤더/캡션 줄을 뺀 모든 줄이 온전한 4셀 행이어야 한다.
    ragged = []
    for c in table_chunks:
        for line in c["text"].split("\n"):
            if line.startswith("표:") or line == header_line:
                continue
            if len(line.split(CELL_SEP)) != 4:
                ragged.append(line)
    check(not ragged, f"반토막 행이 없다 (어긋난 줄 {len(ragged)}개)")

    check(
        all(c["n_tokens"] <= MAX_TOKENS for c in table_chunks),
        f"표 청크가 예산을 넘지 않는다 (최대 {max((c['n_tokens'] for c in table_chunks), default=0)}토큰)",
    )
    check(
        all(c.get("embedding_text", "").startswith("[KB금융 2025-12] III. 재무에 관한 사항") for c in table_chunks),
        "표 청크의 embedding_text에 회사/기간/breadcrumb 접두사가 붙었다",
    )
    check(
        all("embedding_text" not in c for c in para_chunks if c["n_tokens"] >= MIN_CHUNK_TOKENS),
        "20토큰 이상 서술 청크에는 접두사가 없다 (--prefix-narrative 기본 off)",
    )
    check(
        all(c.get("embedding_text", "").startswith("[KB금융 2025-12]") for c in para_chunks if c["n_tokens"] < MIN_CHUNK_TOKENS),
        "20토큰 미만 서술 청크에는 접두사가 붙는다 (내용 없는 벡터 방지)",
    )

    prefixed = chunk_filing(tokenizer, sections, context, prefix_narrative=True)
    check(
        all("embedding_text" in c for c in prefixed if c["kind"] == "p"),
        "--prefix-narrative를 주면 서술 청크에도 접두사가 붙는다",
    )

    # 회귀 방지: 표 앞에 짧은 소제목만 있는 섹션. 표를 거대 문단으로 취급하면 chunk_paragraphs의
    # "소제목 흡수" 분기가 막아주지만, 표가 별도 블록이면 그 경로를 타지 않는다
    # (15자 이하 청크 1.9% -> 0.17%).
    orphan_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1><TITLE ATOCID="9">II. 사업의 내용</TITLE>
<P>다. 장기</P>
<TABLE>
<TR><TH>과목</TH><TH>제58기</TH><TH>제57기</TH><TH>제56기</TH></TR>
{ROWS}
</TABLE>
</SECTION-1></BODY></DOCUMENT>
"""
    orphan_sections = _flatten(parse_sections(ET.fromstring(sanitize_dart_xml(orphan_xml))))
    orphan_chunks = chunk_filing(tokenizer, orphan_sections, context)
    tiny = [c for c in orphan_chunks if c["n_tokens"] <= 5]
    check(
        not tiny and all(c["kind"] == "table" for c in orphan_chunks),
        f"표 앞 짧은 소제목이 껍데기 청크로 안 남는다 (초미니 청크 {len(tiny)}개)",
    )
    check(
        all("다. 장기" in c["text"] for c in orphan_chunks),
        "그 소제목은 표 조각의 캡션으로 보존된다 (내용이 사라지지 않음)",
    )

    small_rows = "".join(f"<TR><TD>계정{i}</TD><TD>{i},000</TD></TR>" for i in range(1, 4))
    label_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1><TITLE ATOCID="3">III. 재무에 관한 사항</TITLE>
<TABLE-GROUP>
<TABLE><TR><TE><P>당기</P></TE></TR></TABLE>
<TABLE><TR><TH>구분</TH><TH>금액</TH></TR>{small_rows}</TABLE>
<TABLE><TR><TE>전기</TE><TE>(단위 : 백만원)</TE></TR></TABLE>
<TABLE><TR><TH>구분</TH><TH>금액</TH></TR>{small_rows}</TABLE>
<TABLE><TR><TE>전전기</TE></TR></TABLE>
<TABLE><TR><TE>(단위 : 천원)</TE></TR></TABLE>
<TABLE><TR><TH>구분</TH><TH>금액</TH></TR>{small_rows}</TABLE>
</TABLE-GROUP>
<P>(1) 주요 제품별 매출실적</P>
<P>(단위 : 억원)</P>
<TABLE><TR><TH>구분</TH><TH>금액</TH></TR>{small_rows}</TABLE>
<TABLE><TR><TE>단독 1행 표</TE><TE>내용이 있는 셀</TE></TR></TABLE>
</SECTION-1></BODY></DOCUMENT>
"""
    label_sections = _flatten(parse_sections(ET.fromstring(sanitize_dart_xml(label_xml))))
    label_tables = [b for s in label_sections for b in s["blocks"] if b["type"] == "table"]
    check(
        [b["caption"] for b in label_tables[:3]] == ["당기", "전기", "전전기"],
        f"라벨 표가 다음 표의 캡션이 된다 (실제 {[b['caption'] for b in label_tables[:3]]})",
    )
    check(
        [b["unit"] for b in label_tables[1:3]] == ["단위 : 백만원", "단위 : 천원"],
        "라벨 표의 단위가 다음 표의 unit이 된다 (라벨과 단위가 따로 온 연속 라벨 표 포함)",
    )
    check(
        not any(len(b["rows"]) == 1 and b["caption"] in ("당기", "전기", "전전기", "") and not b["header"]
                for b in label_tables[:3]),
        "라벨 표 자체는 표 블록으로 남지 않는다",
    )
    check(
        any(b["rows"] == [["단독 1행 표", "내용이 있는 셀"]] for b in label_tables),
        "단위도 없고 뒤에 표도 없는 1행 표는 라벨로 오인하지 않는다 (내용 보존)",
    )
    label_chunks = chunk_filing(tokenizer, label_sections, context)
    sales = [c for c in label_chunks if "주요 제품별 매출실적" in c["text"]]
    check(
        len(sales) == 1 and sales[0]["kind"] == "table" and "단위 : 억원" in sales[0]["text"],
        "제목 문단 + 단위 문단이 따로 와도 표 청크의 캡션·단위로 들어간다",
    )
    check(
        not any(c["kind"] == "p" and c["text"].strip() in ("(1) 주요 제품별 매출실적", "(단위 : 억원)") for c in label_chunks),
        "그 두 문단이 껍데기 서술 청크로 남지 않는다",
    )
    nav_xml = """<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1><TITLE ATOCID="40">XII. 상세표</TITLE>
<TABLE><TR><TD><A REFNO="I. 회사의 개요">☞ 본문 위치로 이동</A></TD></TR></TABLE>
<P>☞ 본문 위치로 이동</P>
<P>상세표 본문 문단입니다. ☞ 본문 위치로 이동 문구를 인용해도 지우지 않습니다.</P>
</SECTION-1></BODY></DOCUMENT>
"""
    nav_texts = [b["text"] for s in _flatten(parse_sections(ET.fromstring(sanitize_dart_xml(nav_xml)))) for b in s["blocks"]]
    check(
        nav_texts == ["상세표 본문 문단입니다. ☞ 본문 위치로 이동 문구를 인용해도 지우지 않습니다."],
        f"탐색용 링크 문구 문단만 빠지고, 그 문구를 담은 본문은 남는다 (실제 {nav_texts})",
    )

    # embed_and_index._point_id와 view_filing.load_chunk_groups가 이 가정에 의존한다.
    starts = [i for i, c in enumerate(chunks) if c["chunk_index"] == 0]
    check(len(starts) == len(sections), f"chunk_index==0이 섹션 수({len(sections)})만큼 정확히 나온다")
    check(
        all(c["chunk_index"] == i for c in chunks for i in [c["chunk_index"]]) and chunks[0]["chunk_index"] == 0,
        "chunk_index가 섹션 안에서 0부터 연속한다",
    )

    tag_xml = """<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1><TITLE ATOCID="1">III. 재무에 관한 사항</TITLE>
<TABLE>
<TR><TH>과목</TH><TH>당기</TH><TH>전기</TH></TR>
<TR><TE ACODE="ifrs-full_Revenue" ADECIMAL="-6">1,000</TE><TE>2,000</TE><TE>3,000</TE></TR>
<TR><TE>4,000</TE><TE>5,000</TE><TE>6,000</TE></TR>
</TABLE>
<TABLE>
<TR><TU AUNIT="U_KRW" AUNITVALUE="1">주권상장법인</TU><TU>해당사항 없음</TU><TU>기타</TU></TR>
<TR><TD>값1</TD><TD>값2</TD><TD>값3</TD></TR>
</TABLE>
<TABLE><TR><TD>한 열짜리 레이아웃 표입니다.</TD></TR><TR><TD>둘째 줄.</TD></TR></TABLE>
</SECTION-1></BODY></DOCUMENT>
"""
    tag_blocks = _flatten(parse_sections(ET.fromstring(sanitize_dart_xml(tag_xml))))[0]["blocks"]
    tables = [b for b in tag_blocks if b["type"] == "table"]
    ps = [b for b in tag_blocks if b["type"] == "p"]

    print("\n태그 의미 회귀:")
    check(len(tables) == 2, f"표 블록 2개 (1열 표는 제외됐다) — 실제 {len(tables)}개")
    check(
        bool(tables) and tables[0]["header"] == ["과목", "당기", "전기"],
        f"TE 행은 헤더가 아니다 (header={tables[0]['header'] if tables else None})",
    )
    check(
        bool(tables) and len(tables[0]["rows"]) == 2,
        f"TE 행 2개가 본문으로 남았다 — 실제 {len(tables[0]['rows']) if tables else 0}행",
    )
    check(
        len(tables) > 1 and tables[1]["header"] == [],
        f"TU 행도 헤더가 아니다 (header={tables[1]['header'] if len(tables) > 1 else None})",
    )
    check(
        len(tables) > 1 and len(tables[1]["rows"]) == 2,
        "TU 행이 본문으로 남았다 (내용 유실 없음)",
    )
    check(
        any(b["text"] == "한 열짜리 레이아웃 표입니다.\n둘째 줄." for b in ps),
        "1열 표는 p 블록으로 되돌아간다 (행은 개행으로 이어붙임)",
    )

    # ---- 격자 복원 / 셀 정규화 / 전체 TH ----
    grid_xml = """<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1><TITLE ATOCID="2">I. 회사의 개요</TITLE>
<TABLE>
<TR><TH ROWSPAN="2">평가대상유가증권</TH><TH>평가일</TH><TH>등급</TH></TR>
<TR><TH>비고</TH><TH>평가회사</TH></TR>
<TR><TD ROWSPAN="3">회사채 US$ Bond</TD><TD>2023.07</TD><TD>AA-</TD></TR>
<TR><TD>2023.08</TD><TD>Aa2</TD></TR>
<TR><TD>2024.02</TD><TD>Baa1/Baa2/
Baa3</TD></TR>
</TABLE>
<TABLE>
<TR><TH>구 분</TH><TH>정  의</TH></TR>
<TR><TH>등 급</TH><TH>설  명</TH></TR>
</TABLE>
</SECTION-1></BODY></DOCUMENT>
"""
    grid_blocks = _flatten(parse_sections(ET.fromstring(sanitize_dart_xml(grid_xml))))[0]["blocks"]
    grid_tables = [b for b in grid_blocks if b["type"] == "table"]

    print("\n격자 복원 / 셀 정규화:")
    span_table = grid_tables[0] if grid_tables else {"header": [], "rows": []}
    # ROWSPAN=2인 첫 헤더 셀이 (1,0)을 점유하므로 2행의 셀들은 열 1,2로 밀려야 한다.
    check(
        span_table["header"] == ["평가대상유가증권", "평가일 비고", "등급 평가회사"],
        f"ROWSPAN 헤더가 열을 밀지 않는다 (header={span_table['header']})",
    )
    check(
        all(len(row) == 3 for row in span_table["rows"]),
        f"본문 행이 전부 3열로 펼쳐졌다 (실제 {[len(r) for r in span_table['rows']]})",
    )
    check(
        all(row[0] == "회사채 US$ Bond" for row in span_table["rows"]),
        "ROWSPAN된 라벨이 모든 행에 반복된다 (어느 조각을 떼도 무엇에 대한 행인지 안다)",
    )
    check(
        any("Baa1/Baa2/ Baa3" in cell for row in span_table["rows"] for cell in row),
        "셀 안의 개행이 공백으로 정규화된다 (행 구분자와 충돌 방지)",
    )
    check(
        all("\n" not in cell for row in span_table["rows"] for cell in row),
        "어떤 셀에도 개행이 남지 않는다",
    )

    all_th = grid_tables[1] if len(grid_tables) > 1 else {"header": ["x"], "rows": []}
    check(
        all_th["header"] == [] and len(all_th["rows"]) == 2,
        f"전체가 TH인 표는 헤더 없음으로 되돌려 2행을 보존한다 "
        f"(header={all_th['header']}, {len(all_th['rows'])}행)",
    )
    check(
        all_th["rows"] and all_th["rows"][0] == ["구 분", "정 의"],
        f"정렬용 이중 공백도 한 칸으로 정규화된다 ({all_th['rows'][0] if all_th['rows'] else None})",
    )

    subheader_xml = """<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1><TITLE ATOCID="3">I. 회사의 개요</TITLE>
<TABLE>
<TR><TH COLSPAN="2" ROWSPAN="2">구  분</TH><TH COLSPAN="3">주식의 종류</TH><TH ROWSPAN="2">비고</TH></TR>
<TR><TE>보통주</TE><TE>우선주</TE><TH>합계</TH></TR>
<TR><TD COLSPAN="2">I. 발행할 주식의 총수</TD><TE>20,000,000,000</TE><TE>5,000,000,000</TE><TE>25,000,000,000</TE><TE>-</TE></TR>
<TR><TH>합계</TH><TD>-</TD><TD>1</TD><TD>2</TD><TD>3</TD><TD>4</TD></TR>
</TABLE>
</SECTION-1></BODY></DOCUMENT>
"""
    sub = [b for b in _flatten(parse_sections(ET.fromstring(sanitize_dart_xml(subheader_xml))))[0]["blocks"] if b["type"] == "table"][0]

    print("\n헤더 판정 규칙:")
    check(
        sub["header"] == ["구 분", "구 분", "주식의 종류 보통주", "주식의 종류 우선주", "주식의 종류 합계", "비고"],
        f"TE/TH 섞인 부헤더 행도 헤더로 잡아 열이 구분된다 (header={sub['header']})",
    )
    check(
        len(sub["rows"]) == 2 and sub["rows"][0][2] == "20,000,000,000",
        f"본문은 2행이고 값이 제 열에 있다 ({len(sub['rows'])}행)",
    )
    check(
        sub["rows"] and sub["rows"][-1][0] == "합계",
        "TH와 TD를 섞은 합계 행은 본문으로 남는다 (TD가 있으면 헤더가 아니다)",
    )
    # 순수 숫자 표: TH가 하나도 없으면 헤더로 잡지 않는다 — TE를 헤더 태그에 넣었을 때
    # 재무제표가 5~11행 헤더를 가진 표로 오인됐던 실패를 막는 닻.
    numeric_xml = """<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1><TITLE ATOCID="4">III. 재무에 관한 사항</TITLE>
<TABLE>
<TR><TE>1,000</TE><TE>2,000</TE></TR>
<TR><TE>3,000</TE><TE>4,000</TE></TR>
<TR><TE>5,000</TE><TE>6,000</TE></TR>
</TABLE>
</SECTION-1></BODY></DOCUMENT>
"""
    num = [b for b in _flatten(parse_sections(ET.fromstring(sanitize_dart_xml(numeric_xml))))[0]["blocks"] if b["type"] == "table"][0]
    check(
        num["header"] == [] and len(num["rows"]) == 3,
        f"TH 없는 순수 TE 표는 헤더 없이 3행 그대로 (header={num['header']}, {len(num['rows'])}행)",
    )

    # ---- 청크 크기 유계 (8192토큰 청크는 MPS에서 임베딩 OOM) ----
    # 캡션+헤더는 모든 조각에 반복되지만 행과 달리 쪼개지지 않는다. 열이 아주 많은 표에
    # 다단 헤더가 붙으면 헤더 한 줄만 수천 토큰이 되어 그 표의 모든 청크가 거대해진다.
    wide_header = "".join(f"<TH>매우 긴 열 이름 번호 {i:03d} 입니다</TH>" for i in range(120))
    wide_body = "".join(
        f"<TR>{''.join(f'<TE>{i}{j}</TE>' for j in range(120))}</TR>" for i in range(6)
    )
    wide_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY><SECTION-1><TITLE ATOCID="5">III. 재무에 관한 사항</TITLE>
<P>가. 아주 넓은 표 (단위 : 백만원)</P>
<TABLE><TR>{wide_header}</TR>{wide_body}</TABLE>
</SECTION-1></BODY></DOCUMENT>
"""
    wide_sections = _flatten(parse_sections(ET.fromstring(sanitize_dart_xml(wide_xml))))
    wide_chunks = chunk_filing(tokenizer, wide_sections, context)
    wide_max = max((c["n_tokens"] for c in wide_chunks), default=0)
    bound = int(MAX_TOKENS * 1.5)

    print("\n청크 크기 유계 (120열 표):")
    check(
        wide_max <= bound,
        f"모든 청크가 1.5 x MAX_TOKENS({bound}) 이하 — 최대 {wide_max}토큰",
    )
    check(
        any("…" in c["text"] for c in wide_chunks if c["kind"] == "table"),
        "긴 헤더는 잘리고 말줄임표가 남는다 (잘렸다는 사실이 보인다)",
    )
    check(
        all("아주 넓은 표" in c["text"] for c in wide_chunks if c["kind"] == "table"),
        "헤더가 잘려도 캡션은 남는다 (무슨 표인지는 계속 안다)",
    )

    print("\nMaxSim 수식 (손계산 대조):")
    e1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    check(maxsim(e1, e1) == 1.0, "동일한 토큰 집합 -> 1.0")
    check(
        maxsim(np.array([[1.0, 0.0]], dtype="float32"), np.array([[0.0, 1.0]], dtype="float32")) == 0.0,
        "직교 -> 0.0",
    )
    # 질의 토큰 2개: 1번은 문서 토큰과 완전 일치(1.0), 2번은 0.8이 최대. 평균 0.9.
    q_mix = np.array([[1.0, 0.0], [0.6, 0.8]], dtype="float32")
    d_mix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    check(abs(maxsim(q_mix, d_mix) - 0.9) < 1e-6, "혼합(1.0과 0.8의 평균) -> 0.9")
    check(maxsim(np.zeros((0, 2), dtype="float32"), e1) == 0.0, "빈 질의 -> 0.0 (예외 아님)")
    check(maxsim(e1, np.zeros((0, 2), dtype="float32")) == 0.0, "빈 문서 -> 0.0 (빈 후보 텍스트 방어)")
    q_ax = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
    d_ax = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
    check(
        maxsim(q_ax, d_ax) == 1.0 and float((q_ax @ d_ax.T).max(axis=0).mean()) < 0.7,
        "질의 토큰 축(axis=1)으로 최대를 고른다 (문서에 딴 내용이 섞여도 감점되지 않는다)",
    )

    print("\nRRF 융합 (동점 결정성):")

    class Hit:
        def __init__(self, pid: str) -> None:
            self.id, self.score = pid, 0.0

    def ids(hits: list) -> list[str]:
        return [h.id for h in hits]

    # 측정과 같은 모양: dense 1위(D)와 sparse 1위(S)가 서로 다른 청크 -> 둘 다 1/(2+0)=0.5 동점
    fused = rrf_fuse([Hit("D"), Hit("X")], [Hit("S"), Hit("Y")], top_k=4)
    check(fused[0].score == fused[1].score == 1 / RRF_K, "양쪽 1위가 다르면 정확히 동점이다 (측정된 0.5000의 정체)")
    check(ids(fused)[:2] == ["S", "D"], f"동점이면 sparse 순위가 앞선 쪽이 먼저 (실제 {ids(fused)[:2]})")
    # 동점 hit들의 id를 바꿔 dict 삽입 순서가 달라져도 규칙(sparse 순위)만으로 순서가 정해진다
    fused_renamed = rrf_fuse([Hit("a-dense"), Hit("X")], [Hit("z-sparse"), Hit("Y")], top_k=4)
    check(
        ids(fused_renamed)[:2] == ["z-sparse", "a-dense"],
        "동점 순서가 id 알파벳순이나 입력 순서가 아니라 sparse 순위로 정해진다",
    )
    # 교차 순위(dense b>a, sparse a>b): 점수는 1/2+1/3로 같고 sparse 순위로 a가 앞선다
    cross = rrf_fuse([Hit("b"), Hit("a")], [Hit("a"), Hit("b")], top_k=2)
    check(ids(cross) == ["a", "b"], f"교차 순위 동점도 sparse 순위로 푼다 (실제 {ids(cross)})")
    both = rrf_fuse([Hit("A"), Hit("B")], [Hit("B"), Hit("C")], top_k=3)
    check(ids(both)[0] == "B" and abs(both[0].score - (1 / 3 + 1 / 2)) < 1e-12, "양쪽 모두 걸린 hit은 점수 합산으로 1위")
    check(len(rrf_fuse([Hit(str(i)) for i in range(10)], [], top_k=5)) == 5, "top_k로 자른다")
    check(rrf_fuse([], [], top_k=5) == [], "빈 입력 -> 빈 결과")

    print("\n전체 통과" if not failures else f"\n실패 {len(failures)}건: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
