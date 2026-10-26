from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import fact_store as fs

CTX = "{p}_ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis_ifrs-full_{s}Member"


def te(code: str, period: str, text: str, dec: str = "-6", neg: str = "N", scope: str = "Consolidated", extra: str = "") -> str:
    ctx = CTX.format(p=period, s=scope) + extra
    return f'<TE ACODE="{code}" ACONTEXT="{ctx}" ADECIMAL="{dec}" ANEGATED="{neg}">{text}</TE>'


XML = f"""<?xml version="1.0" encoding="utf-8"?>
<DOCUMENT><BODY>
<SECTION-1><TITLE ATOCID="1">III. 재무에 관한 사항</TITLE>
<SECTION-2><TITLE ATOCID="2">2. 연결재무제표</TITLE>
<P>연결 재무상태표

제 57 기          2025.12.31 현재</P>
<TABLE>
<TR><TH></TH><TH>제 57 기</TH><TH>제 56 기</TH></TR>
<TR><TE ENG="Assets">자산</TE><TE ACODE="ifrs-full_AssetsAbstract"></TE><TE ACODE="ifrs-full_AssetsAbstract"></TE></TR>
<TR><TE ENG="&#12288;Current assets">유동자산</TE>{te("ifrs-full_CurrentAssets", "CFY2025eFY", "247,684,612")}{te("ifrs-full_CurrentAssets", "PFY2024eFY", "227,062,266")}</TR>
<TR><TE ENG="Total assets">자산총계</TE>{te("ifrs-full_Assets", "CFY2025eFY", "500")}{te("ifrs-full_Assets", "PFY2024eFY", "450")}</TR>
<TR><TE ENG="Total liabilities">부채총계</TE>{te("ifrs-full_Liabilities", "CFY2025eFY", "200")}{te("ifrs-full_Liabilities", "PFY2024eFY", "180")}</TR>
<TR><TE ENG="Total equity">자본총계</TE>{te("ifrs-full_Equity", "CFY2025eFY", "300")}{te("ifrs-full_Equity", "PFY2024eFY", "270")}</TR>
</TABLE>
<P>연결 포괄손익계산서</P>
<TABLE>
<TR><TE ENG="Interest expense">이자비용</TE>{te("ifrs-full_InterestExpense", "CFY2025dFY", "(16,083,007)", neg="Y")}{te("ifrs-full_InterestExpense", "PFY2024dFY", "(17,664,671)", neg="Y")}</TR>
<TR><TE ENG="Other loss">기타손실</TE>{te("dart_OtherLosses", "CFY2025dFY", "(1,000)")}{te("dart_OtherLosses", "PFY2024dFY", "2,000", neg="Y")}</TR>
<TR><TE ENG="Revenue">매출액 (주30)</TE>{te("ifrs-full_Revenue", "CFY2025dFY", "4,556,971,695,180", dec="0")}{te("ifrs-full_Revenue", "PFY2024dFY", "-", dec="0")}</TR>
</TABLE>
<P>연결 현금흐름표</P>
<TABLE>
<TR><TE>투자활동현금흐름</TE>{te("ifrs-full_CashFlowsFromUsedInInvestingActivities", "CFY2025dFY", "(56,341,712)")}</TR>
<TR><TE>장기금융상품의 처분</TE>{te("dart_ProceedsFromSalesOfLongTermFinancialInstruments", "CFY2025dFY", "4,519,179")}</TR>
<TR><TE>장기금융상품의 취득</TE>{te("dart_PurchaseOfLongTermFinancialInstruments", "CFY2025dFY", "(13,533,556)", neg="Y")}</TR>
<TR><TE>유형자산의 처분</TE>{te("ifrs-full_ProceedsFromSalesOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "CFY2025dFY", "149,828")}</TR>
<TR><TE>유형자산의 취득</TE>{te("ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "CFY2025dFY", "(47,522,179)", neg="Y")}</TR>
<TR><TE>기타투자활동</TE>{te("dart_OtherInvestingActivities", "CFY2025dFY", "45,016")}</TR>
</TABLE>
<P>자본변동표</P>
<TABLE>
<TR><TE>이익잉여금</TE>{te("ifrs-full_Equity", "CFY2025eFY", "120", extra="_ifrs-full_ComponentsOfEquityAxis_ifrs-full_RetainedEarningsMember")}</TR>
<TR><TE>별도 자본</TE>{te("ifrs-full_Equity", "CFY2025eFY", "250", scope="Separate")}</TR>
<TR><TE>깨진 컨텍스트</TE><TE ACODE="x_Y" ACONTEXT="CFY2025eFY_weird">1</TE></TR>
</TABLE>
</SECTION-2></SECTION-1></BODY></DOCUMENT>
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  [{'OK  ' if cond else 'FAIL'}] {label}")
        if not cond:
            failures.append(label)

    print("ACONTEXT 해석:")
    c = fs.parse_context(CTX.format(p="CFY2025eFY", s="Consolidated"))
    check((c.fiscal_year, c.period_role, c.period_type, c.scope, c.dims) == (2025, "당기", "instant", "연결", ""),
          "당기 시점형 연결 본표 -> (2025, 당기, instant, 연결, '')")
    c = fs.parse_context(CTX.format(p="BPFY2023dFY", s="Separate") + "_ifrs-full_ComponentsOfEquityAxis_ifrs-full_RetainedEarningsMember")
    check((c.period_role, c.period_type, c.scope) == ("전전기", "duration", "별도"), "전전기 기간형 별도")
    check(c.dims == "ifrs-full_ComponentsOfEquityAxis=ifrs-full_RetainedEarningsMember", f"추가 차원은 축=멤버로 (실제 {c.dims!r})")
    for bad in ("CFY2025eFY_weird", "XFY2025eFY_a_b_c_d", "CFY2025eFY_ifrs-full_OtherAxis_ifrs-full_ConsolidatedMember"):
        try:
            fs.parse_context(bad)
            check(False, f"가정이 깨진 컨텍스트는 거부: {bad}")
        except ValueError:
            check(True, f"가정이 깨진 컨텍스트는 거부: {bad}")

    print("\n값 해석 (부호·스케일):")
    check(fs.parse_value("247,684,612", "-6") == 247_684_612e6, "DEC=-6 표시값 x 10^6 (백만원 -> 원)")
    check(fs.parse_value("4,556,971,695,180", "0") == 4_556_971_695_180, "DEC=0은 원 단위 그대로")
    check(fs.parse_value("(1,000)", "-6") == -1_000e6, "괄호 -> 음수")
    check(fs.parse_value("12.5", "INF") == 12.5, "INF는 정확값 — 스케일 안 곱함")
    check(fs.parse_value("-", "-6") is None and fs.parse_value("", "0") is None, "대시·빈칸 -> None")
    check(fs.parse_value("해당없음", "-6") is None, "비숫자 -> None")

    print("\n부호 규칙 (ANEGATED는 값에 반영하지 않는다):")
    neg_rows, _ = fs.extract_facts(XML, {"company": "삼성전자", "period": "2025.12", "rcept_no": "R2025"})
    by = {(r[3], r[4]): r for r in neg_rows}
    ie = by[("ifrs-full_InterestExpense", 2025)]
    check(ie[9] == -16_083_007e6 and ie[12] == 1, "괄호 + NEGATED=Y -> 표시 그대로 음수, negated는 기록만")
    ol = by[("dart_OtherLosses", 2024)]
    check(ol[9] == 2_000e6 and ol[12] == 1, "숫자 + NEGATED=Y -> 표시 그대로 양수")
    # 투자활동현금흐름 소계 = 항목 합이 표시값으로 성립한다. ANEGATED를 반영하면 깨진다.
    inv = [r for r in neg_rows if r[16] == "연결 현금흐름표" and r[4] == 2025]
    sub = next(r for r in inv if r[3] == "ifrs-full_CashFlowsFromUsedInInvestingActivities")
    items = [r for r in inv if r is not sub]
    check(sum(r[9] for r in items) == sub[9], f"투자활동 소계 = 항목 합 (표시값 규칙, 실제 {sum(r[9] for r in items):,.0f} vs {sub[9]:,.0f})")
    flipped = sum(-r[9] if r[12] else r[9] for r in items)
    check(flipped != sub[9], "ANEGATED를 반영하면 소계가 안 맞는다")

    print("\n적재 + 조회:")
    entry = {"company": "삼성전자", "period": "2025.12", "rcept_no": "R2025"}
    rows, skipped = fs.extract_facts(XML, entry)
    check(skipped == 1, f"깨진 컨텍스트 1개는 건너뛰고 센다 (실제 {skipped})")
    check(all(r[3] != "ifrs-full_AssetsAbstract" for r in rows), "ACONTEXT 없는 ACODE 셀(제목 행)은 fact가 아니다")

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.sqlite"
        import sqlite3
        con = sqlite3.connect(db)
        con.executescript(fs.SCHEMA)
        con.executemany(f"INSERT INTO facts VALUES ({','.join('?' * 19)})", rows)
        # 다음 해 보고서 흉내: 2025 값이 전기로 다시 나온다(같은 값) — get_series가 최신 보고서를 고르는지
        con.execute(
            f"INSERT INTO facts VALUES ({','.join('?' * 19)})",
            ("삼성전자", "2026.12", "R2026", "ifrs-full_Equity", 2025, "전기", "instant", "연결", "", 301e6,
             "301", "-6", 0, "자본총계", "Total equity", "2. 연결재무제표", "연결 재무상태표", 3, 1),
        )
        # 더 최신 보고서의 **주석** 표에 같은 fact가 1,000배 작게(천원 표시, DEC 동일) 있다 — NAVER
        # 현금및현금성자산 사례. 최신 보고서라도 주석 값보다 본표 값을 골라야 한다.
        con.execute(
            f"INSERT INTO facts VALUES ({','.join('?' * 19)})",
            ("삼성전자", "2027.12", "R2027", "ifrs-full_Equity", 2025, "전전기", "instant", "연결", "", 301e3,
             "301", "-3", 0, "자본총계", "", "3. 연결재무제표 주석", "금융자산의 범주별 공시", 9, 0),
        )
        con.commit(); con.close()

        found = fs.find_concepts("삼성전자", "자본총계", db)
        check(bool(found) and found[0]["concept"] == "ifrs-full_Equity", "한글 '자본총계' -> ifrs-full_Equity")
        check(any(f["concept"] == "ifrs-full_Equity" for f in fs.find_concepts("삼성전자", "Total equity", db)), "영문 라벨로도 찾는다")

        series = fs.get_series("삼성전자", "ifrs-full_Equity", "연결", db)
        check([s["fiscal_year"] for s in series] == [2024, 2025], "연도별 시리즈, 오름차순")
        check(series[-1]["value"] == 301e6 and series[-1]["source_report"] == "2026.12",
              "본표 값이 여러 보고서에 있으면 최신 보고서 값(재작성 반영)")
        check(series[-1]["value"] != 301e3,
              "더 최신 보고서라도 주석 표 값(1,000배 틀린 단위)보다 본표 값을 고른다")
        check(all(s["value"] != 120e6 for s in series), "추가 차원(자본변동표 구성요소) 값은 시리즈에 안 섞인다")
        sep = fs.get_series("삼성전자", "ifrs-full_Equity", "별도", db)
        check([s["value"] for s in sep] == [250e6], "별도는 별도로 — 연결과 안 섞인다")

        rev = fs.query_facts("삼성전자", concept="ifrs-full_Revenue", fiscal_year=2025, dims="", db_path=db)
        check(len(rev) == 1 and rev[0]["value"] == 4_556_971_695_180, "원 단위 표시 매출액 조회")
        check(rev[0]["table_title"] == "연결 포괄손익계산서" and rev[0]["section"] == "2. 연결재무제표",
              f"출처(섹션·표 제목) 보존 (실제 {rev[0]['section']!r} / {rev[0]['table_title']!r})")
        bs = fs.query_facts("삼성전자", concept="ifrs-full_CurrentAssets", fiscal_year=2025, db_path=db)
        check(bs[0]["table_title"] == "연결 재무상태표", "여러 줄 제목 문단은 첫 줄만 표 제목으로")
        cur = fs.query_facts("삼성전자", label="유동자산", fiscal_year=2025, db_path=db)
        check(bool(cur), "라벨 부분일치 조회")
        lbl_en = [r for r in rows if r[3] == "ifrs-full_CurrentAssets"][0][14]
        check(lbl_en == "Current assets", f"ENG 라벨의 전각 공백 들여쓰기 제거 (실제 {lbl_en!r})")
        dims_all = fs.query_facts("삼성전자", concept="ifrs-full_Equity", fiscal_year=2025, db_path=db)
        check(any(r["dims"] for r in dims_all), "dims=None이면 추가 차원 값도 나온다")

    print("\n전체 통과" if not failures else f"\n실패 {len(failures)}건: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
