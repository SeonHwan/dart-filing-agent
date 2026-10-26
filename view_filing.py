from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path

from dart_sections import table_caption_line
from parse_filing import parse_one

DATA_DIR = Path(__file__).parent / "data"
DART_VIEWER = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"


def list_filings() -> list[tuple[str, str, Path]]:
    """data/{회사}/{기간}/ 을 훑어서 (회사, 기간, 경로) 목록을 만든다."""
    found = []
    for company_dir in sorted(p for p in DATA_DIR.glob("*") if p.is_dir()):
        for period_dir in sorted(p for p in company_dir.glob("*") if p.is_dir()):
            found.append((company_dir.name, period_dir.name, period_dir))
    return found


def load_sections(filing_dir: Path) -> list[dict]:
    sections_path = filing_dir / "sections.json"
    if sections_path.exists():
        return json.loads(sections_path.read_text(encoding="utf-8"))

    candidates = [p for p in filing_dir.glob("*.xml") if "_" not in p.stem]
    if not candidates:
        raise SystemExit(f"{filing_dir}에 본문 XML도 sections.json도 없습니다.")
    print(f"sections.json이 없어서 {candidates[0].name}을 즉석 파싱합니다...")
    return parse_one(candidates[0])


def load_chunk_groups(filing_dir: Path, n_sections: int) -> list[list[dict]] | None:
    path = filing_dir / "chunks.jsonl"
    if not path.exists():
        return None

    groups: list[list[dict]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["chunk_index"] == 0 or not groups:
                groups.append([])
            groups[-1].append(chunk)

    if len(groups) != n_sections:
        print(f"경고: 섹션 {n_sections}개 vs 청크 그룹 {len(groups)}개 — 대응이 어긋나서 청크 뷰를 생략합니다.")
        return None
    return groups


def render_block(block: dict) -> str:
    if block["type"] == "p":
        return f"<p>{html.escape(block['text'])}</p>"

    out = []
    caption = table_caption_line(block)
    if caption != "표":  # 캡션도 단위도 없으면 굳이 붙이지 않는다
        out.append(f'<div class="tcap">{html.escape(caption)}</div>')
    out.append("<table>")
    if block.get("header"):
        out.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in block["header"]) + "</tr>")
    for row in block["rows"]:
        out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
    out.append("</table>")
    return "".join(out)


def render_html(company: str, period: str, sections: list[dict], chunk_groups: list[list[dict]] | None, rcept_no: str | None) -> str:
    toc, body = [], []

    for i, section in enumerate(sections):
        path = section["path"]
        title = path[-1] if path else f"(제목 없음 #{i})"
        n_chunks = len(chunk_groups[i]) if chunk_groups else 0
        n_tables = sum(1 for b in section["blocks"] if b["type"] == "table")
        counts = f'{len(section["blocks"]) - n_tables}문단' + (f" · 표{n_tables}" if n_tables else "")

        toc.append(
            f'<a class="lv{section["level"]}" href="#s{i}" data-title="{html.escape(title)}">'
            f'{html.escape(title)}<span class="n">{counts}'
            f'{f" · {n_chunks}청크" if chunk_groups else ""}</span></a>'
        )

        paras = "".join(render_block(b) for b in section["blocks"])
        chunks_html = ""
        if chunk_groups:
            cards = []
            for c in chunk_groups[i]:
                cards.append(
                    f'<div class="chunk"><div class="badge">#{c["chunk_index"] + 1}/{c["n_chunks"]}'
                    f' · {c["n_tokens"]}토큰 · {"표" if c.get("kind") == "table" else "글"}'
                    f'</div><pre>{html.escape(c["text"])}</pre></div>'
                )
            chunks_html = f'<div class="chunks">{"".join(cards)}</div>'

        body.append(
            f'<section id="s{i}">'
            f'<div class="crumb">{html.escape(" › ".join(path))}'
            f'<span class="meta">level {section["level"]} · ATOCID {section["atocid"]}</span></div>'
            f'<div class="cols"><div class="pane paras">{paras}</div>{chunks_html}</div>'
            f'</section>'
        )

    dart_link = (
        f'<a class="ext" href="{DART_VIEWER.format(rcept_no=rcept_no)}" target="_blank">DART 원문 ↗</a>'
        if rcept_no else ""
    )
    chunk_toggle = '<button data-mode="chunks">청크</button><button data-mode="both">나란히</button>' if chunk_groups else ""

    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(company)} {html.escape(period)} — 사업보고서 뷰어</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --dim:#6b7280; --line:#e5e7eb; --accent:#2563eb; --card:#f9fafb; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#111318; --fg:#e6e8eb; --dim:#9aa1ab; --line:#2a2f39; --accent:#6ea8fe; --card:#1a1e26; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; display:flex; font:14px/1.65 -apple-system,"Segoe UI","Malgun Gothic",sans-serif;
         background:var(--bg); color:var(--fg); }}
  #toc {{ width:320px; flex:0 0 320px; height:100vh; overflow:auto; border-right:1px solid var(--line); padding:12px; }}
  #toc input {{ width:100%; padding:6px 8px; margin-bottom:8px; border:1px solid var(--line);
                border-radius:6px; background:var(--bg); color:var(--fg); }}
  #toc a {{ display:block; padding:3px 6px; color:var(--fg); text-decoration:none; border-radius:4px; }}
  #toc a:hover {{ background:var(--card); }}
  #toc a.lv2 {{ padding-left:20px; }} #toc a.lv3 {{ padding-left:36px; font-size:13px; }}
  #toc a.lv1 {{ font-weight:600; margin-top:8px; }}
  #toc .n {{ float:right; color:var(--dim); font-size:11px; }}
  main {{ flex:1; height:100vh; overflow:auto; padding:0 28px 120px; }}
  header {{ position:sticky; top:0; background:var(--bg); padding:14px 0; border-bottom:1px solid var(--line);
            display:flex; gap:8px; align-items:center; flex-wrap:wrap; z-index:5; }}
  header h1 {{ font-size:16px; margin:0 auto 0 0; }}
  header button, .ext {{ font:inherit; padding:4px 10px; border:1px solid var(--line); border-radius:6px;
                         background:var(--bg); color:var(--fg); cursor:pointer; text-decoration:none; }}
  header button.on {{ border-color:var(--accent); color:var(--accent); }}
  section {{ padding-top:22px; }}
  .crumb {{ font-weight:600; color:var(--accent); border-bottom:1px solid var(--line); padding-bottom:5px;
            margin-bottom:10px; display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }}
  .crumb .meta {{ font-weight:400; font-size:11px; color:var(--dim); margin-left:auto; }}
  .cols {{ display:flex; gap:18px; align-items:flex-start; }}
  .pane {{ flex:1; min-width:0; }}
  .tcap {{ font-size:12px; color:var(--dim); margin:10px 0 -4px; }}
  table {{ border-collapse:collapse; margin:10px 0; font-size:12px; display:block; overflow-x:auto; max-width:100%; }}
  th, td {{ border:1px solid var(--line); padding:3px 7px; text-align:left; white-space:nowrap; }}
  th {{ background:var(--card); }}
  .chunks {{ flex:1; min-width:0; }}
  .chunk {{ border:1px solid var(--line); border-radius:8px; margin-bottom:10px; background:var(--card); }}
  .chunk .badge {{ font-size:11px; color:var(--dim); padding:5px 9px; border-bottom:1px solid var(--line); }}
  .chunk pre {{ margin:0; padding:9px; white-space:pre-wrap; word-break:break-word; font-size:12px; }}
  body.mode-paras .chunks, body.mode-chunks .paras {{ display:none; }}
</style>
<nav id="toc">
  <input id="filter" placeholder="목차 검색...">
  {"".join(toc)}
</nav>
<main>
  <header>
    <h1>{html.escape(company)} · {html.escape(period)} <span style="color:var(--dim);font-weight:400">
      섹션 {len(sections)}개{f" · 청크 {sum(len(g) for g in chunk_groups)}개" if chunk_groups else ""}</span></h1>
    <button data-mode="paras" class="on">본문</button>{chunk_toggle}{dart_link}
  </header>
  {"".join(body)}
</main>
<script>
  document.body.className = "mode-paras";
  document.querySelectorAll("header button").forEach(b => b.onclick = () => {{
    document.body.className = "mode-" + b.dataset.mode;
    document.querySelectorAll("header button").forEach(x => x.classList.toggle("on", x === b));
  }});
  filter.oninput = () => {{
    const q = filter.value.trim().toLowerCase();
    document.querySelectorAll("#toc a").forEach(a => {{
      a.style.display = !q || a.dataset.title.toLowerCase().includes(q) ? "" : "none";
    }});
  }};
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="DART 사업보고서 파싱·청킹 결과 뷰어")
    ap.add_argument("company", nargs="?", help="회사명 (예: 삼성전자)")
    ap.add_argument("period", nargs="?", help="기간 디렉터리명 (예: 2025-03)")
    ap.add_argument("--list", action="store_true", help="사용 가능한 필링 목록만 출력")
    ap.add_argument("--open", action="store_true", help="렌더링 후 브라우저로 열기")
    args = ap.parse_args()

    filings = list_filings()
    if not filings:
        raise SystemExit(f"{DATA_DIR}에 필링이 없습니다.")

    if args.list or not args.company:
        print(f"사용 가능한 필링 ({len(filings)}건):")
        for company, period, path in filings:
            marks = [name for name in ("sections.json", "chunks.jsonl") if (path / name).exists()]
            print(f"  {company:<20} {period}   {', '.join(marks) or '원본 XML만'}")
        return

    matches = [f for f in filings if f[0] == args.company and (args.period is None or f[1] == args.period)]
    if not matches:
        raise SystemExit(f"'{args.company} {args.period or ''}'에 해당하는 필링이 없습니다. --list로 확인하세요.")
    company, period, filing_dir = matches[0]
    if len(matches) > 1:
        print(f"'{company}'에 기간이 여러 개라 {period}을 씁니다 (기간을 직접 지정하면 그것으로).")

    sections = load_sections(filing_dir)
    chunk_groups = load_chunk_groups(filing_dir, len(sections))
    rcept_no = next((p.stem for p in filing_dir.glob("*.xml") if "_" not in p.stem), None)

    out_path = filing_dir / "view.html"
    out_path.write_text(render_html(company, period, sections, chunk_groups, rcept_no), encoding="utf-8")
    n_chunks = sum(len(g) for g in chunk_groups) if chunk_groups else 0
    print(f"{company} {period}: 섹션 {len(sections)}개, 청크 {n_chunks}개 -> {out_path}")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
