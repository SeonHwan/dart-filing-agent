from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from dart_sections import CELL_SEP, UNIT_RE, _strip_unit, serialize_table, table_caption_line

DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_MODEL = "BAAI/bge-m3"
MAX_TOKENS = 512
OVERLAP_TOKENS = 64
MIN_CHUNK_TOKENS = 20
# 설계상 표 청크의 상한은 고정부(<=MAX_TOKENS/2) + 행묶음(<=budget) = 1.5 * MAX_TOKENS다.
# 이걸 넘으면 아직 유계가 아닌 경로가 남아 있다는 뜻이라 main()에서 경고한다.
OVERSIZE_WARN_TOKENS = MAX_TOKENS * 2


def _split_long_paragraph(tokenizer, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    # offset_mapping(토큰 경계 -> 원문 문자 위치)으로 원문을 직접 슬라이싱한다.
    # tokenizer.decode()로 되돌리면 bge-m3(XLM-R/SentencePiece)의 공백 정규화 때문에 개행이
    # 공백으로 뭉개진다("...백만원)\n매출액..." -> "...백만원) 매출액...").
    # 표를 행 단위로 개행 직렬화해둔 자리에서 하필 행 구분이 사라지는 게 제일 아프다.
    # offsets는 fast 토크나이저 전용 기능이고 bge-m3는 XLMRobertaTokenizerFast라 쓸 수 있다
    # (use_fast=False로 로드하면 깨짐).
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n = len(offsets)
    stride = max(1, max_tokens - overlap_tokens)
    pieces: list[str] = []
    start = 0
    # 마지막 조각을 "끝에서 거꾸로 max_tokens"로 채우지 않는다. 남은 꼬리가 짧으면 바로 앞 조각과
    # 최대 98%까지 겹쳐 사실상 동일한 청크가 두 개 생기고(5000토큰 문단에서 마지막 두 조각이
    # (4480,4992)/(4488,5000)으로 504토큰 동일), 검색 top-k 슬롯과 임베딩 계산이 낭비된다.
    # 초미니 꼬리 조각은 아래 _merge_tiny_chunks가 이웃에 흡수시킨다.
    while start < n:
        end = min(start + max_tokens, n)
        # 남은 꼬리가 overlap_tokens 이하로 짧으면 별도 조각으로 떼지 않고 이번 조각에
        # 흡수시킨다. 안 그러면 그 꼬리는 내용 대부분이 앞 조각과 겹친 껍데기 청크가 된다
        # (5018토큰 문단에서 마지막 조각 90토큰 중 64토큰이 겹침).
        if 0 < n - end <= overlap_tokens:
            end = n
        pieces.append(text[offsets[start][0] : offsets[end - 1][1]])
        if end == n:
            break
        start += stride
    return pieces


def _merge_tiny_chunks(tokenizer, chunks: list[str], min_tokens: int = MIN_CHUNK_TOKENS) -> list[str]:
    if len(chunks) <= 1:
        return chunks

    merged = list(chunks)
    # 청크 토큰 수를 배치로 한 번에 재둔다(fast 토크나이저는 리스트를 받으면 Rust에서
    # 병렬 처리하므로, 루프마다 encode를 부르는 것보다 훨씬 빠름). 병합이 일어난 자리만
    # 다시 정확히 재계산한다 — 두 청크의 토큰 수 합은 구분자 "\n\n" 때문에 실제와
    # 어긋날 수 있어서 더하기로 때우지 않는다.
    counts = [len(ids) for ids in tokenizer(merged, add_special_tokens=False)["input_ids"]]
    i = 0
    while i < len(merged):
        if counts[i] >= min_tokens:
            i += 1
            continue
        if i + 1 < len(merged):
            merged[i + 1] = merged[i] + "\n\n" + merged[i + 1]
            merged.pop(i)
            counts.pop(i)
            counts[i] = len(tokenizer.encode(merged[i], add_special_tokens=False))
        elif i > 0:
            merged[i - 1] = merged[i - 1] + "\n\n" + merged[i]
            merged.pop(i)
            counts.pop(i)
            counts[i - 1] = len(tokenizer.encode(merged[i - 1], add_special_tokens=False))
        else:
            i += 1
    return merged


def chunk_paragraphs(tokenizer, paragraphs: list[str], max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    if not paragraphs:
        return []
    counts = [len(ids) for ids in tokenizer(paragraphs, add_special_tokens=False)["input_ids"]]

    chunks: list[str] = []
    buf: list[str] = []
    buf_counts: list[int] = []

    def flush():
        # buf/buf_counts를 리바인딩하지 않고 제자리에서 비우므로 nonlocal이 필요 없다.
        if buf:
            chunks.append("\n\n".join(buf))
        buf.clear()
        buf_counts.clear()

    for para, para_tokens in zip(paragraphs, counts):
        if para_tokens > max_tokens:
            if buf and sum(buf_counts) <= overlap_tokens:
                # buf에 남은 게 "다. 장기"처럼 소제목 하나뿐인 짧은 문단이면, 그걸 그냥
                # flush해서 혼자 초미니 청크로 만들지 않고 뒤에 오는 거대 문단 앞에 붙여서
                # 같이 슬라이딩 윈도우로 분할한다 — 소제목만 남는 청크가 임베딩 검색에서
                # 무관한 질문에도 비정상적으로 높은 유사도로 뜬다
                # (KB금융 재무제표 주석에서 373/19950개 청크가 15자 이하).
                heading = "\n\n".join(buf)
                # heading을 이미 떠놨으니 flush()로 청크를 만들지 않고 버퍼만 비운다.
                buf.clear()
                buf_counts.clear()
                chunks.extend(_split_long_paragraph(tokenizer, heading + "\n\n" + para, max_tokens, overlap_tokens))
            else:
                flush()
                chunks.extend(_split_long_paragraph(tokenizer, para, max_tokens, overlap_tokens))
            continue

        if buf and sum(buf_counts) + para_tokens > max_tokens:
            last_para, last_tokens = buf[-1], buf_counts[-1]
            flush()
            if last_tokens <= overlap_tokens:
                buf.append(last_para)
                buf_counts.append(last_tokens)

        buf.append(para)
        buf_counts.append(para_tokens)

    flush()
    return _merge_tiny_chunks(tokenizer, chunks)


def _truncate_tokens(tokenizer, text: str, max_tokens: int) -> str:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) <= max_tokens:
        return text
    return text[: offsets[max_tokens - 1][1]] + " …"


def chunk_table(tokenizer, block: dict, max_tokens: int = MAX_TOKENS) -> list[str]:
    if not block["rows"]:
        return []

    caption_line = table_caption_line(block)
    header_line = CELL_SEP.join(block["header"]) if block.get("header") else ""
    fixed_text = "\n".join(line for line in (caption_line, header_line) if line)

    # **고정부에 반드시 상한을 둬야 한다.** 캡션+헤더는 모든 조각에 반복되지만 행과 달리
    # 쪼개지지 않으므로, 길면 그 표의 모든 청크가 통째로 거대해진다. 열이 아주 많은 표
    # (20열 이상이 289개)에 다단 헤더가 붙으면 헤더 한 줄이 수천 토큰이 되고, 그러면
    # 8192토큰(bge-m3 한도) 입력이 만들어져 MPS에서 임베딩이 8GiB 할당을 시도하다
    # OOM이 난다. 절반으로 제한하면 청크는 항상
    # fixed(<=max/2) + 행묶음(<=budget) <= 1.5 * max_tokens로 유계가 된다.
    fixed_text = _truncate_tokens(tokenizer, fixed_text, max_tokens // 2)

    row_lines = [CELL_SEP.join(row) for row in block["rows"]]
    measured = tokenizer([fixed_text, *row_lines], add_special_tokens=False)["input_ids"]
    fixed_tokens = len(measured[0])
    row_counts = [len(ids) for ids in measured[1:]]

    # 고정부가 예산을 거의 다 먹는 표에서도 최소 한 행은 담아야 하므로 하한을 둔다.
    budget = max(max_tokens - fixed_tokens, MIN_CHUNK_TOKENS)

    def build(lines: list[str]) -> str:
        return "\n".join([fixed_text, *lines]) if fixed_text else "\n".join(lines)

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for line, count in zip(row_lines, row_counts):
        if count > budget:
            if buf:
                chunks.append(build(buf))
                buf, buf_tokens = [], 0
            chunks.extend(build([piece]) for piece in _split_long_paragraph(tokenizer, line, budget, 0))
            continue
        if buf and buf_tokens + count > budget:
            chunks.append(build(buf))
            buf, buf_tokens = [], 0
        buf.append(line)
        buf_tokens += count

    if buf:
        chunks.append(build(buf))
    return chunks


def _covered_by_table(tokenizer, text: str, block: dict) -> bool:
    if len(tokenizer.encode(text, add_special_tokens=False)) > MIN_CHUNK_TOKENS:
        return False
    rest = _strip_unit(text)
    if not rest:
        return bool(block.get("unit")) and UNIT_RE.search(text) is not None
    return rest in (block.get("caption") or "")


def _context_prefix(context: dict, path: list[str]) -> str:
    head = f"[{context['company']} {context['period']}]"
    return f"{head} {' > '.join(path)}".strip()


def chunk_filing(tokenizer, sections: list[dict], context: dict, prefix_narrative: bool = False) -> list[dict]:
    records = []
    for section in sections:
        prefix = _context_prefix(context, section["path"])
        # (텍스트, 종류) 순서쌍으로 모은다. 블록 순서를 지켜야 문서 안 위치가 보존된다.
        pieces: list[tuple[str, str]] = []
        para_buf: list[str] = []

        def flush_paragraphs():
            if para_buf:
                pieces.extend((text, "p") for text in chunk_paragraphs(tokenizer, para_buf))
                para_buf.clear()

        for block in section["blocks"]:
            if block["type"] == "p":
                para_buf.append(block["text"])
                continue

            # 표 바로 앞의 짧은 소제목·단위 문단은 독립 청크로 내보내지 않는다. 그 텍스트는 이 표의
            # caption/unit으로 모든 표 조각에 이미 들어가 있으므로, 따로 내보내면 "다. 장기"처럼
            # 4글자짜리 껍데기 청크가 된다. 표를 거대 문단으로 취급하면 chunk_paragraphs의
            # "소제목 흡수" 분기가 이를 막지만(15자 이하 청크 1.9% -> 0.17%),
            # 표가 별도 블록이면 그 경로를 타지 않는다.
            #
            # `(1) 주요 제품별 매출실적` + `(단위 : 억원)`처럼 제목과 단위가 두 문단으로 나뉜 경우도
            # 있다(표 사이에 낀 20토큰 미만 문단 866개 중 91%가 바로 뒤가 표). 그래서 버퍼
            # **끝에서부터** 표의 캡션·단위가 이미 담고 있는 짧은 문단을 걷어낸다.
            while para_buf and _covered_by_table(tokenizer, para_buf[-1], block):
                para_buf.pop()

            flush_paragraphs()
            pieces.extend((text, "table") for text in chunk_table(tokenizer, block))
        flush_paragraphs()

        if not pieces:
            continue

        texts = [text for text, _ in pieces]
        token_counts = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
        for idx, ((text, kind), n_tokens) in enumerate(zip(pieces, token_counts)):
            record = {
                "atocid": section["atocid"],
                "level": section["level"],
                "path": section["path"],
                "kind": kind,
                "chunk_index": idx,
                "n_chunks": len(pieces),
                "text": text,
                "n_tokens": n_tokens,
            }
            # 20토큰 미만 서술 청크에는 --prefix-narrative 없이도 접두사를 붙인다.
            # `- 해당사항 없음`처럼 섹션 내용이 진짜 그것뿐인 청크는 지울 수 없는데(답이 "없음"인
            # 질문의 근거다), 본문만 임베딩하면 내용이 없는 벡터가 되어 무관한 질의의 dense 상위에
            # 올라온다(`sk-risk` dump 22·24·26·28위). 목차 경로를 붙이면 "무엇이
            # 해당사항 없는지"가 벡터에 들어간다. 긴 서술 청크는 회귀 위험 때문에 여전히 옵트인이다.
            if kind == "table" or prefix_narrative or n_tokens < MIN_CHUNK_TOKENS:
                record["embedding_text"] = f"{prefix}\n{text}"
            records.append(record)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="섹션 블록을 임베딩용 청크로 재분할")
    ap.add_argument(
        "--prefix-narrative",
        action="store_true",
        help="서술 청크에도 회사/기간/breadcrumb 문맥 접두사를 붙인다(기본은 표 청크만)",
    )
    args = ap.parse_args()

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    for entry in manifest:
        filing_dir = DATA_DIR / entry["company"] / entry["period"].replace(".", "-")
        sections = json.loads((filing_dir / "sections.json").read_text(encoding="utf-8"))
        context = {"company": entry["company"], "period": entry["period"]}
        chunks = chunk_filing(tokenizer, sections, context, prefix_narrative=args.prefix_narrative)

        out_path = filing_dir / "chunks.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                meta = {
                    "company": entry["company"],
                    "corp_code": entry["corp_code"],
                    "period": entry["period"],
                    "rcept_no": entry["rcept_no"],
                }
                f.write(json.dumps({**meta, **chunk}, ensure_ascii=False) + "\n")

        n_tokens = [c["n_tokens"] for c in chunks]
        avg = sum(n_tokens) / len(n_tokens) if n_tokens else 0
        n_table = sum(1 for c in chunks if c["kind"] == "table")
        print(
            f"{entry['company']} {entry['period']}: 청크 {len(chunks)}개"
            f"(표 {n_table}개/서술 {len(chunks) - n_table}개), 평균 {avg:.0f}토큰,"
            f" 최대 {max(n_tokens, default=0)}토큰 -> {out_path}"
        )

        over = [c for c in chunks if c["n_tokens"] > OVERSIZE_WARN_TOKENS]
        if over:
            worst = max(over, key=lambda c: c["n_tokens"])
            print(
                f"  !! 경고: {len(over)}개 청크가 {OVERSIZE_WARN_TOKENS}토큰 초과."
                f" 최대 {worst['n_tokens']}토큰 (kind={worst['kind']}) @ {' > '.join(worst['path'])}"
            )


if __name__ == "__main__":
    main()
