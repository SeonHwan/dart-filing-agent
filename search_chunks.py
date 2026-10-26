from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from bge_m3 import BgeM3Encoder
from embed_and_index import COLLECTION_NAME
from eval_retrieval import _search, rerank_hits

load_dotenv()

# 모듈 최상단에서 os.environ[...]으로 필수화하면 import만으로 KeyError가 난다
# (CI에서 import만으로 깨지지 않게). docstring과
# .env.example이 약속한 기본값을 코드에도 반영한다.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--company", help="회사명으로 결과를 좁힘 (예: 삼성전자)")
    parser.add_argument("--collection", default=COLLECTION_NAME, help=f"조회할 컬렉션 (기본 {COLLECTION_NAME})")
    parser.add_argument("--mode", choices=("hybrid", "dense", "sparse"), default="hybrid", help="검색 방식")
    parser.add_argument("--rerank", action="store_true", help="colbert(MaxSim)로 후보 재정렬")
    parser.add_argument("--rerank-depth", type=int, default=50, help="리랭킹할 후보 수 (기본 50)")
    args = parser.parse_args()

    encoder = BgeM3Encoder()
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False, timeout=60)
    fetch_k = max(args.top_k, args.rerank_depth) if args.rerank else args.top_k
    hits = _search(
        client, encoder.encode_query(args.query), args.company, fetch_k, args.collection, args.mode
    )
    if args.rerank:
        hits = rerank_hits(encoder, args.query, hits, args.top_k)

    # 점수의 의미가 모드마다 다르다: dense/sparse는 유사도, hybrid는 RRF 점수(순위의 역수 합),
    # --rerank를 주면 MaxSim이다. 모드 간에 점수를 직접 비교하면 안 된다.
    note = f" / rerank(depth={args.rerank_depth})" if args.rerank else ""
    print(f"mode={args.mode}{note} / 컬렉션 {args.collection} / {len(hits)}건\n")
    for hit in hits:
        p = hit.payload
        kind = {"table": "표", "p": "글"}.get(p.get("kind"), "?")
        # 리랭킹했으면 검색 단계의 원래 순위를 같이 보여준다(무엇이 끌어올려졌는지 보이게).
        orig = f" (검색 {p['_orig_rank']}위)" if "_orig_rank" in p else ""
        print(
            f"[{hit.score:.4f}][{kind}]{orig} {p['company']} {p['period']} > {' > '.join(p['path'])}"
            f" (청크 {p['chunk_index']}/{p['n_chunks']})"
        )
        print(p["text"][:200].replace("\n", " "))
        print("---")


if __name__ == "__main__":
    main()
