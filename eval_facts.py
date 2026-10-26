from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from fact_store import DB_PATH, find_concepts, get_series

QUERIES_PATH = Path(__file__).parent / "eval_facts.json"


NOTE_REF_RE = re.compile(r"\s*\((주|단위)[^)]*\)\s*$")


def _pick_concept(candidates: list[dict], keyword: str) -> dict | None:
    if not candidates:
        return None

    def bare(label: str) -> str:
        return NOTE_REF_RE.sub("", label or "").strip()

    exact = [c for c in candidates if bare(c["row_label"]) == keyword]
    starts = [c for c in candidates if bare(c["row_label"]).startswith(keyword)]
    return (exact or starts or candidates)[0]


def evaluate(q: dict, db_path: Path) -> dict:
    candidates = find_concepts(q["company"], q["concept_keyword"], db_path)
    chosen = _pick_concept(candidates, q["concept_keyword"])
    result = {"id": q["id"], "candidates": candidates, "chosen": chosen, "got": None, "row": None}
    if chosen is None:
        result["verdict"] = "계정 못 찾음"
        return result
    series = {r["fiscal_year"]: r for r in get_series(q["company"], chosen["concept"], q["scope"], db_path)}
    row = series.get(q["fiscal_year"])
    result["row"] = row
    if row is None:
        result["verdict"] = "연도 없음"
        return result
    result["got"] = row["value"]
    expected = q["expect"]["value"]
    # 원 단위 정수 비교. float 저장이라 1원 미만 오차만 허용한다.
    if abs(row["value"] - expected) < 1:
        result["verdict"] = "정답"
    elif row["value"] == -expected:
        result["verdict"] = "값 틀림(부호)"
    elif expected and round(abs(row["value"] / expected)) in (1_000, 1_000_000) or (
        row["value"] and round(abs(expected / row["value"])) in (1_000, 1_000_000)
    ):
        result["verdict"] = "값 틀림(스케일)"
    else:
        # 계정 선택이 틀렸는지 값 계산이 틀렸는지 가른다: 정답 값이 다른 후보 계정에 있으면 선택 문제다.
        for other in candidates:
            if other is chosen:
                continue
            alt = {r["fiscal_year"]: r["value"] for r in get_series(q["company"], other["concept"], q["scope"], db_path)}
            if abs(alt.get(q["fiscal_year"], float("nan")) - expected) < 1:
                result["verdict"] = f"계정 오선택(정답은 {other['concept']})"
                break
        else:
            result["verdict"] = "값 틀림"
    return result


UNIT_SCALE = {"원": 1, "천원": 10**3, "백만원": 10**6}


def verify_answers(queries: list[dict]) -> None:
    manifest = json.loads((DB_PATH.parent / "manifest.json").read_text(encoding="utf-8"))
    by_report = {(e["company"], e["period"]): e for e in manifest}
    xml_cache: dict[tuple, str] = {}
    ok = 0
    for q in queries:
        exp = q["expect"]
        key = (q["company"], exp["report"])
        if key not in xml_cache:
            e = by_report[key]
            xml_cache[key] = (DB_PATH.parent / e["company"] / e["period"].replace(".", "-") / f"{e['rcept_no']}.xml").read_text(encoding="utf-8")
        xml = xml_cache[key]
        row_found = unit_found = None
        for m in re.finditer(r"<TR(?=[\s>]).*?</TR>", xml, re.S):
            cells = [re.sub(r"<[^>]+>|\s+", " ", c).strip()
                     for c in re.findall(r"<T[DEHU](?=[\s>])[^>]*>(.*?)</T[DEHU]>", m.group(0), re.S)]
            if not cells or not cells[0].lstrip("　 ").startswith(exp["row_label"]) or exp["display"] not in cells[1:]:
                continue
            row_found = cells[0]
            in_label = re.search(r"단위\s*[::]\s*(백만원|천원|원)", cells[0])
            above = re.findall(r"단위\s*[::]\s*(백만원|천원|원)", xml[max(0, m.start() - 60000) : m.start()])
            unit_found = in_label.group(1) if in_label else (above[-1] if above else None)
            break
        t = exp["display"].replace(",", "")
        arith = int(t.strip("()")) * UNIT_SCALE[exp["unit"]] * (-1 if t.startswith("(") else 1) == exp["value"]
        checks = {"행": row_found is not None, "단위": unit_found == exp["unit"], "환산": arith}
        passed = all(checks.values())
        ok += passed
        detail = "  ".join(f"{k}{'✓' if v else '✗'}" for k, v in checks.items())
        extra = f"  (원문 단위 {unit_found})" if row_found and unit_found != exp["unit"] else ""
        print(f"  [{'OK  ' if passed else 'FAIL'}] {q['id']:<26} {detail}  {row_found or '행 없음'}{extra}")
    print(f"원문 대조 {ok}/{len(queries)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="정형 스토어 평가 — 정확한 값이 정답인 질문")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--verify-answers", action="store_true", help="정답 값 자체를 원본 XML과 대조(fact_store 미사용)")
    args = ap.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if args.verify_answers:
        verify_answers(queries)
        return
    results = [evaluate(q, args.db) for q in queries]

    print(f"정형 스토어 평가 — 질문 {len(queries)}개 ({args.db.name})\n{'=' * 96}")
    print(f"{'id':<26}{'판정':<18}{'기대값':>24}{'조회값':>24}  선택 계정")
    print("-" * 96)
    for q, r in zip(queries, results):
        got = f"{r['got']:,.0f}" if r["got"] is not None else "-"
        concept = r["chosen"]["concept"] if r["chosen"] else "-"
        print(f"{q['id']:<26}{r['verdict']:<18}{q['expect']['value']:>24,}{got:>24}  {concept[:40]}")
        if args.verbose:
            print(f"    질문: {q['query']}  (키워드 {q['concept_keyword']!r}, {q['fiscal_year']} {q['scope']})")
            for c in r["candidates"][:4]:
                mark = "->" if c is r["chosen"] else "  "
                print(f"    {mark} {c['n']:>3}회  {c['concept'][:55]:<55} {c['row_label'][:24]}")
            if r["row"]:
                print(f"    출처: {r['row']['source_report']} 보고서 > {r['row']['table_title']} > {r['row']['row_label']}"
                      f"  (표시 {r['row']['raw_text']})")
            print(f"    정답 근거: {q['expect']['source']}")

    n_ok = sum(r["verdict"] == "정답" for r in results)
    print("-" * 96)
    print(f"정답 {n_ok}/{len(results)} ({n_ok / len(results):.0%})")
    wrong = [r for r in results if r["verdict"] != "정답"]
    if wrong:
        print("오답 유형: " + ", ".join(f"{r['id']}={r['verdict']}" for r in wrong))


if __name__ == "__main__":
    main()
