from __future__ import annotations

import argparse
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dart_sections import TABLE_CELL_TAGS
from dart_xml_sanitize import sanitize_dart_xml

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "facts.sqlite"

# 조각[1..4]는 전 fact 공통의 연결/별도 축이다(62,717/62,717). 이 축 이름이 다르게 나오면
# 스키마 가정이 깨진 것이므로 적재를 멈춘다(조용히 섞이는 것보다 낫다).
SCOPE_AXIS = "ifrs-full_ConsolidatedAndSeparateFinancialStatementsAxis"
SCOPE_MEMBERS = {"ifrs-full_ConsolidatedMember": "연결", "ifrs-full_SeparateMember": "별도"}
PERIOD_RE = re.compile(r"^(CFY|PFY|BPFY)(\d{4})([de])FY$")
PERIOD_ROLE = {"CFY": "당기", "PFY": "전기", "BPFY": "전전기"}
# 본표 섹션. 주석 표에도 같은 계정 코드가 붙는데 **표시 단위가 달라도 ADECIMAL이 같은** 경우가 있다 —
# NAVER 현금및현금성자산이 재무상태표에 `5,984,018,799,741`(원), 금융자산 범주 주석에 `5,984,018,800`
# (천원 표시)으로 나오는데 둘 다 DEC=0이다. 한 보고서 안 충돌 197건 중 157건이 이 천/백만 배 차이였다.
# 그래서 같은 fact가 여러 표에 있으면 본표 값을 쓴다.
PRIMARY_SECTION_RE = re.compile(r"^\d+\.\s*(연결)?재무제표$")
NUM_RE = re.compile(r"^\(?\s*-?[\d,]+(\.\d+)?\s*\)?$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    company     TEXT NOT NULL,   -- manifest의 회사명 (예: 삼성전자)
    period      TEXT NOT NULL,   -- manifest의 보고 기간 (예: 2025-12) — 어느 보고서에서 왔나
    rcept_no    TEXT NOT NULL,
    concept     TEXT NOT NULL,   -- ACODE (예: ifrs-full_Equity)
    fiscal_year INTEGER NOT NULL,-- 값이 속한 회계연도 (ACONTEXT 조각[0]의 연도)
    period_role TEXT NOT NULL,   -- 당기/전기/전전기 (그 보고서 기준)
    period_type TEXT NOT NULL,   -- duration(손익·현금흐름) / instant(재무상태 잔액)
    scope       TEXT NOT NULL,   -- 연결 / 별도
    dims        TEXT NOT NULL,   -- 추가 차원 "축=멤버;..." (없으면 '' = 본표 값)
    value       REAL,            -- 원 단위로 환산·부호 반영한 값 (파싱 불가면 NULL)
    raw_text    TEXT NOT NULL,   -- 보고서에 표시된 그대로
    decimals    TEXT,            -- ADECIMAL 원문
    negated     INTEGER NOT NULL,-- ANEGATED=Y면 1. 원본 속성 기록일 뿐 value 계산에는 안 쓴다
    row_label   TEXT NOT NULL,   -- 같은 행의 라벨 셀 텍스트(한글 계정명)
    row_label_en TEXT,           -- 라벨 셀의 ENG 속성
    section     TEXT NOT NULL,   -- 가장 가까운 섹션 제목 (예: 2. 연결재무제표)
    table_title TEXT NOT NULL,   -- 표 앞 문단 첫 줄 (예: 연결 재무상태표)
    table_idx   INTEGER NOT NULL,-- 문서 안 TABLE 순번 — 같은 표의 fact를 묶는 키
    primary_stmt INTEGER NOT NULL -- 본표(재무제표 섹션, 주석 아님)면 1. 조회는 본표 값을 우선한다
);
CREATE INDEX IF NOT EXISTS ix_facts_lookup ON facts (company, concept, scope, dims);
CREATE INDEX IF NOT EXISTS ix_facts_label  ON facts (company, row_label);
CREATE INDEX IF NOT EXISTS ix_facts_table  ON facts (rcept_no, table_idx);
"""


@dataclass
class Context:
    fiscal_year: int
    period_role: str
    period_type: str
    scope: str
    dims: str


def parse_context(ctx: str) -> Context:
    """ACONTEXT를 기간/연결·별도/추가 차원으로 푼다. 가정이 깨지면 ValueError(모듈 docstring 참고)."""
    parts = ctx.split("_")
    if len(parts) < 5 or (len(parts) - 1) % 4 != 0:
        raise ValueError(f"ACONTEXT 조각 수가 1+4k가 아님: {ctx}")
    m = PERIOD_RE.match(parts[0])
    if not m:
        raise ValueError(f"기간 조각을 못 읽음: {parts[0]}")
    role, year, kind = m.groups()
    # 조각을 (접두사_이름) 쌍으로 다시 붙인다: ifrs-full + ConsolidatedAndSeparate...Axis
    pairs = ["_".join(parts[i : i + 2]) for i in range(1, len(parts), 2)]
    axes, members = pairs[0::2], pairs[1::2]
    if axes[0] != SCOPE_AXIS or members[0] not in SCOPE_MEMBERS:
        raise ValueError(f"첫 축이 연결/별도 축이 아님: {ctx}")
    dims = ";".join(f"{a}={mem}" for a, mem in zip(axes[1:], members[1:]))
    return Context(int(year), PERIOD_ROLE[role], "duration" if kind == "d" else "instant", SCOPE_MEMBERS[members[0]], dims)


def parse_value(text: str, decimals: str | None) -> float | None:
    t = text.strip().replace("−", "-")
    if not t or not NUM_RE.match(t):
        return None
    paren = t.startswith("(") and t.endswith(")")
    digits = t.strip("()").strip()
    value = float(digits.replace(",", ""))
    if paren:
        value = -abs(value)
    # decimals ≤ 0 이면 표시 단위가 10^(-decimals) 원이다(-6 = 백만원). INF·양수는 정확값.
    if decimals not in (None, "", "INF"):
        try:
            d = int(decimals)
        except ValueError:
            d = 0
        if d < 0:
            value *= 10 ** (-d)
    return value


def _text(elem: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return re.sub(r"\s+", " ", line).strip()[:80]
    return ""


def extract_facts(xml_text: str, entry: dict, skipped_samples: list | None = None) -> tuple[list[tuple], int]:
    root = ET.fromstring(sanitize_dart_xml(xml_text))
    parents = {child: parent for parent in root.iter() for child in parent}

    def section_of(elem: ET.Element) -> str:
        cur = parents.get(elem)
        while cur is not None:
            if cur.tag.startswith("SECTION-") and cur.find("TITLE") is not None:
                return _text(cur.find("TITLE"))
            cur = parents.get(cur)
        return ""

    def title_of(table: ET.Element) -> str:
        parent = parents.get(table)
        if parent is None:
            return ""
        siblings = list(parent)
        for prev in reversed(siblings[: siblings.index(table)]):
            # 표 제목 문단은 "연결 재무상태표\n\n제 57 기 ..." 형태라 개행을 살려 첫 줄만 쓴다.
            raw = "".join(prev.itertext())
            if raw.strip():
                return _first_line(raw)
        return ""

    rows: list[tuple] = []
    skipped = 0
    for table_idx, table in enumerate(root.iter("TABLE")):
        section = table_title = None
        for tr in table.iter("TR"):
            cells = [c for c in tr if c.tag in TABLE_CELL_TAGS]
            facts = [c for c in cells if c.get("ACONTEXT") and c.get("ACODE")]
            if not facts:
                continue
            if section is None:
                section, table_title = section_of(table), title_of(table)
            # 라벨 셀 = 같은 행에서 ACONTEXT가 없는 첫 번째 비어있지 않은 셀(19,435개 전부 TE+ENG).
            label = next((c for c in cells if not c.get("ACONTEXT") and _text(c)), None)
            row_label = _text(label) if label is not None else ""
            row_label_en = label.get("ENG", "").replace("　", " ").strip() if label is not None else ""
            for cell in facts:
                try:
                    ctx = parse_context(cell.get("ACONTEXT"))
                except ValueError as exc:
                    skipped += 1
                    if skipped_samples is not None and len(skipped_samples) < 5:
                        skipped_samples.append(f"{row_label[:20]!r}: {exc}")
                    continue
                raw = _text(cell)
                negated = cell.get("ANEGATED") == "Y"
                rows.append((
                    entry["company"], entry["period"], entry["rcept_no"], cell.get("ACODE"),
                    ctx.fiscal_year, ctx.period_role, ctx.period_type, ctx.scope, ctx.dims,
                    parse_value(raw, cell.get("ADECIMAL")), raw, cell.get("ADECIMAL"),
                    int(negated), row_label, row_label_en, section, table_title, table_idx,
                    int(bool(PRIMARY_SECTION_RE.match(section))),
                ))
    return rows, skipped


def build(db_path: Path = DB_PATH) -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    db_path.unlink(missing_ok=True)  # 파생 데이터라 매번 새로 만든다(원본은 XML).
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    total = 0
    for entry in manifest:
        xml_path = DATA_DIR / entry["company"] / entry["period"].replace(".", "-") / f"{entry['rcept_no']}.xml"
        samples: list[str] = []
        rows, skipped = extract_facts(xml_path.read_text(encoding="utf-8"), entry, samples)
        con.executemany(f"INSERT INTO facts VALUES ({','.join('?' * 19)})", rows)
        total += len(rows)
        note = f", 형식 불일치로 건너뜀 {skipped}" if skipped else ""
        print(f"{entry['company']} {entry['period']}: fact {len(rows):,}개{note}")
        for sample in samples:
            print(f"    건너뜀 예: {sample[:160]}")
    con.commit()
    con.close()
    print(f"완료: {total:,}개 -> {db_path}")


# ------------------------------------------------------------------ 조회 함수 (Phase 3 도구의 원형)


@contextmanager
def _connect(db_path: Path):
    if not db_path.exists():
        raise SystemExit(f"{db_path}가 없습니다. 먼저: python fact_store.py build")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def find_concepts(company: str, keyword: str, db_path: Path = DB_PATH, limit: int = 20) -> list[dict]:
    with _connect(db_path) as con:
        like = f"%{keyword}%"
        rows = con.execute(
            """SELECT concept, row_label, row_label_en, COUNT(*) AS n
               FROM facts
               WHERE company = ? AND dims = '' AND (row_label LIKE ? OR row_label_en LIKE ? OR concept LIKE ?)
               GROUP BY concept, row_label ORDER BY n DESC LIMIT ?""",
            (company, like, like, like, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_series(company: str, concept: str, scope: str = "연결", db_path: Path = DB_PATH) -> list[dict]:
    with _connect(db_path) as con:
        rows = con.execute(
            """SELECT fiscal_year, period_type, value, raw_text, decimals, period AS source_report,
                      row_label, section, table_title, primary_stmt
               FROM facts
               WHERE company = ? AND concept = ? AND scope = ? AND dims = '' AND value IS NOT NULL
               ORDER BY fiscal_year, primary_stmt DESC, period DESC, table_idx""",
            (company, concept, scope),
        ).fetchall()
        series: dict[int, dict] = {}
        for r in rows:
            series.setdefault(r["fiscal_year"], dict(r))  # 정렬상 첫 번째가 우선순위 1위
        return [series[y] for y in sorted(series)]


def query_facts(company: str, concept: str | None = None, fiscal_year: int | None = None,
                scope: str | None = None, dims: str | None = None, label: str | None = None,
                db_path: Path = DB_PATH, limit: int = 50) -> list[dict]:
    """조건으로 fact를 직접 조회한다. dims=''이면 본표 값만, None이면 전부."""
    with _connect(db_path) as con:
        where, args = ["company = ?"], [company]
        for col, val in (("concept", concept), ("fiscal_year", fiscal_year), ("scope", scope), ("dims", dims)):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        if label:
            where.append("row_label LIKE ?")
            args.append(f"%{label}%")
        rows = con.execute(
            f"""SELECT concept, fiscal_year, period_role, period_type, scope, dims, value, raw_text,
                       row_label, section, table_title, period AS source_report
                FROM facts WHERE {' AND '.join(where)}
                ORDER BY fiscal_year DESC, period DESC LIMIT ?""",
            (*args, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def check(db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as con:
        # ## 왜 두 단계로 나눠 세는가
        # (회사, 계정, 연도, 연결/별도)로만 묶어 "여러 보고서에 나왔는데 값이 다른 그룹"을 세면
        # **한 보고서 안에서** 같은 계정이 여러 번 나오는 경우(재무상태표의 자본총계 vs 자본변동표
        # 합계 행, 요약재무정보 vs 본표)도 불일치로 잡힌다(이 방식은 83.3%가 나왔다).
        # 그래서 먼저 보고서 안에서의 중복을 따로 세고, 보고서 간 비교는 **보고서마다 값이 하나로
        # 정해지는 그룹**만 대상으로 한다. 불일치 예시도 찍는다 — 비율만으로는 원인을 가를 수 없다.
        # row_label로는 묶지 않는다 — "현금및현금성자산 (주4,29)"처럼 주석 번호가 해마다 바뀐다.
        key = "company, concept, fiscal_year, period_type, scope"
        print("== 1a. 한 보고서 안에서 같은 fact가 서로 다른 값으로 나오는 그룹 (본표 값)")
        r = con.execute(
            f"""SELECT COUNT(*) AS groups, SUM(nvals > 1) AS multi FROM (
                    SELECT COUNT(DISTINCT value) AS nvals FROM facts
                    WHERE dims = '' AND value IS NOT NULL GROUP BY {key}, period)"""
        ).fetchone()
        print(f"  (보고서, fact) 그룹 {r['groups']:,}개 중 한 보고서 안에서 값이 여럿인 그룹 {r['multi']:,}개")
        for x in con.execute(
            f"""SELECT {key}, period, GROUP_CONCAT(DISTINCT raw_text) AS vals,
                       GROUP_CONCAT(DISTINCT table_title) AS tables
                FROM facts WHERE dims = '' AND value IS NOT NULL
                GROUP BY {key}, period HAVING COUNT(DISTINCT value) > 1 LIMIT 5"""
        ):
            print(f"    {x['company']} {x['period']} {x['concept'][:45]} {x['fiscal_year']} {x['scope']}"
                  f"  값[{x['vals'][:60]}]  표[{(x['tables'] or '')[:60]}]")

        print("\n== 1b. 보고서 간 일치율 (보고서 안에서 값이 하나로 정해지는 그룹만)")
        r = con.execute(
            f"""WITH per_report AS (
                    SELECT {key}, period, MIN(value) AS value FROM facts
                    WHERE dims = '' AND value IS NOT NULL
                    GROUP BY {key}, period HAVING COUNT(DISTINCT value) = 1)
                SELECT COUNT(*) AS groups, SUM(nvals = 1) AS consistent FROM (
                    SELECT COUNT(DISTINCT value) AS nvals FROM per_report
                    GROUP BY {key} HAVING COUNT(DISTINCT period) > 1)"""
        ).fetchone()
        if r["groups"]:
            print(f"  여러 보고서에 나온 그룹 {r['groups']:,}개 중 값이 같은 그룹 {r['consistent']:,}개"
                  f" ({r['consistent'] / r['groups']:.1%})")
        for x in con.execute(
            f"""WITH per_report AS (
                    SELECT {key}, period, MIN(value) AS value, MIN(raw_text) AS raw FROM facts
                    WHERE dims = '' AND value IS NOT NULL
                    GROUP BY {key}, period HAVING COUNT(DISTINCT value) = 1)
                SELECT {key}, GROUP_CONCAT(period || '=' || raw, '  ') AS by_report FROM per_report
                GROUP BY {key} HAVING COUNT(DISTINCT period) > 1 AND COUNT(DISTINCT value) > 1 LIMIT 8"""
        ):
            print(f"    {x['company']} {x['concept'][:50]} {x['fiscal_year']} {x['scope']}  {x['by_report'][:80]}")

        print("\n== 2. 자산 = 부채 + 자본 (연결, 본표)")
        rows = con.execute(
            """SELECT company, fiscal_year,
                      MAX(CASE WHEN concept='ifrs-full_Assets' THEN value END) AS a,
                      MAX(CASE WHEN concept='ifrs-full_Liabilities' THEN value END) AS l,
                      MAX(CASE WHEN concept='ifrs-full_Equity' THEN value END) AS e
               FROM facts WHERE scope='연결' AND dims='' AND period_type='instant'
               GROUP BY company, fiscal_year ORDER BY company, fiscal_year"""
        ).fetchall()
        for r in rows:
            if None in (r["a"], r["l"], r["e"]):
                print(f"  {r['company']:<10} {r['fiscal_year']}  (자산/부채/자본 중 누락)")
                continue
            gap = r["a"] - (r["l"] + r["e"])
            print(f"  {r['company']:<10} {r['fiscal_year']}  자산 {r['a']:>22,.0f}  부채+자본 {r['l'] + r['e']:>22,.0f}"
                  f"  차이 {gap:,.0f} {'OK' if abs(gap) < 1e6 else '<-- 불일치'}")

        # 부호 규칙의 검증. 자산=부채+자본은 부호 반전 항목이 없어 부호를 못 가린다 — 현금흐름표는
        # 유입(+)과 유출(-)이 한 소계에 섞이므로 부호가 하나라도 틀리면 합이 어긋난다. ANEGATED를
        # 값에 반영하면 이 검사가 실패한다(2025.12 보고서 30/30 실패).
        print("\n== 3. 투자활동현금흐름 = 그 아래 항목 합 (연결·별도, 본표) — 부호 규칙 검증")
        ok = fail = 0
        shown = 0
        groups = con.execute(
            """SELECT DISTINCT company, period, scope, fiscal_year FROM facts
               WHERE dims = '' AND table_title LIKE '%현금흐름표%' ORDER BY company, period, scope, fiscal_year"""
        ).fetchall()
        for g in groups:
            rows = con.execute(
                """SELECT concept, value, decimals FROM facts
                   WHERE company=? AND period=? AND scope=? AND fiscal_year=? AND dims='' AND period_type='duration'
                   AND table_title LIKE '%현금흐름표%' ORDER BY rowid""",
                (g["company"], g["period"], g["scope"], g["fiscal_year"]),
            ).fetchall()
            concepts = [r["concept"] for r in rows]
            inv, fin = "ifrs-full_CashFlowsFromUsedInInvestingActivities", "ifrs-full_CashFlowsFromUsedInFinancingActivities"
            if inv not in concepts or fin not in concepts or concepts.index(fin) < concepts.index(inv):
                continue
            i, j = concepts.index(inv), concepts.index(fin)
            sub, items = rows[i], rows[i + 1 : j]
            unit = 10 ** max(0, -int(sub["decimals"])) if sub["decimals"] not in (None, "", "INF") else 1
            gap = (sub["value"] or 0) - sum((r["value"] or 0) for r in items)
            if abs(gap) <= unit * len(items):  # 표시 단위의 반올림 오차는 항목 수만큼 허용
                ok += 1
            else:
                fail += 1
                if shown < 5:
                    shown += 1
                    print(f"  불일치 {g['company']} {g['period']} {g['scope']} {g['fiscal_year']}  차이 {gap:,.0f}")
        print(f"  성립 {ok} / 불일치 {fail}")


def main() -> None:
    ap = argparse.ArgumentParser(description="4단계 정형 스토어 — XBRL fact 적재·조회")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="data/의 XML 전부에서 facts.sqlite를 새로 만든다")
    sub.add_parser("check", help="적재 결과 자체 점검(보고서 간 일치, 자산=부채+자본, 부호 표본)")
    p = sub.add_parser("find", help="계정 찾기: 한글/영문 라벨이나 코드로")
    p.add_argument("company"); p.add_argument("keyword")
    p = sub.add_parser("series", help="한 계정의 연도별 값")
    p.add_argument("company"); p.add_argument("concept"); p.add_argument("--scope", default="연결", choices=("연결", "별도"))
    p = sub.add_parser("query", help="조건 조회")
    p.add_argument("company"); p.add_argument("--concept"); p.add_argument("--year", type=int)
    p.add_argument("--scope", choices=("연결", "별도")); p.add_argument("--label")
    p.add_argument("--all-dims", action="store_true", help="주석 등 추가 차원 값도 포함(기본은 본표만)")
    args = ap.parse_args()

    if args.cmd == "build":
        build()
    elif args.cmd == "check":
        check()
    elif args.cmd == "find":
        for r in find_concepts(args.company, args.keyword):
            print(f"{r['n']:>4}  {r['concept']:<60} {r['row_label']}  / {r['row_label_en']}")
    elif args.cmd == "series":
        for r in get_series(args.company, args.concept, args.scope):
            print(f"{r['fiscal_year']}  {r['value']:>24,.0f} 원   (표시 {r['raw_text']}, decimals={r['decimals']},"
                  f" 출처 {r['source_report']} 보고서 > {r['table_title']})")
    elif args.cmd == "query":
        for r in query_facts(args.company, args.concept, args.year, args.scope, None if args.all_dims else "", args.label):
            val = f"{r['value']:,.0f}" if r["value"] is not None else "-"
            print(f"{r['fiscal_year']} {r['scope']} {r['period_role']:<3} {val:>24}  {r['row_label'][:24]:<24} {r['concept']}"
                  + (f"  [{r['dims']}]" if r["dims"] else ""))


if __name__ == "__main__":
    main()
