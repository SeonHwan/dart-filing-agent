from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from bge_m3 import BgeM3Encoder
from embed_and_index import COLLECTION_NAME, DENSE_VECTOR, SPARSE_VECTOR, to_sparse_vector
from fusion import rrf_fuse

load_dotenv()

QUERIES_PATH = Path(__file__).parent / "eval_queries.json"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
RECALL_AT = (1, 5, 10, 50)
TABLE_WIDTH = 89
TABLE_MARK = " | "  # 표 청크의 셀 구분자 (dart_sections.CELL_SEP)


def _is_table(payload: dict) -> bool:
    kind = payload.get("kind")
    if kind:
        return kind == "table"
    return TABLE_MARK in (payload.get("text") or "")


def _matches(payload: dict, expect: dict) -> bool:
    breadcrumb = " > ".join(payload.get("path") or [])
    for needle in expect.get("path_contains") or []:
        if needle not in breadcrumb:
            return False
    path_any = expect.get("path_any") or []
    if path_any and not any(n in breadcrumb for n in path_any):
        return False
    text_needles = expect.get("text_contains") or []
    if text_needles and not any(n in payload.get("text", "") for n in text_needles):
        return False
    return True


def _is_labeled(expect: dict) -> bool:
    return bool(expect.get("path_contains") or expect.get("path_any") or expect.get("text_contains"))


def _hit_density(hits: list, expect: dict) -> float:
    if not hits:
        return 0.0
    return sum(1 for h in hits if _matches(h.payload, expect)) / len(hits)


def _collection_report(client: QdrantClient, collection: str, companies: list[str]) -> bool:
    if not client.collection_exists(collection):
        existing = [c.name for c in client.get_collections().collections]
        raise SystemExit(
            f"컬렉션 '{collection}'이 없습니다.\n"
            f"  현재 있는 컬렉션: {', '.join(existing) or '(없음)'}\n"
            f"  먼저 적재하세요: python embed_and_index.py --collection {collection}\n"
            f"  옛 컬렉션을 재려면: python eval_retrieval.py --collection <이름> --mode dense"
        )

    total = client.count(collection).count
    print(f"컬렉션 {collection}: 포인트 {total:,}개")
    missing = []
    for company in sorted(set(companies)):
        n = client.count(
            collection,
            count_filter=Filter(must=[FieldCondition(key="company", match=MatchValue(value=company))]),
        ).count
        print(f"  {company:<16} {n:>8,}")
        if n == 0:
            missing.append(company)
    if missing:
        print(f"  !! 경고: {', '.join(missing)}의 포인트가 0개입니다 — 적재가 덜 끝났습니다.")
        print("     이 상태의 지표는 다른 컬렉션과 비교할 수 없습니다(질문이 결과 0건이 됨).")
    return not missing


def _query_filter(company: str | None, chunk_kind: str | None) -> Filter | None:
    must = []
    if company:
        must.append(FieldCondition(key="company", match=MatchValue(value=company)))
    if chunk_kind:
        must.append(FieldCondition(key="kind", match=MatchValue(value=chunk_kind)))
    return Filter(must=must) if must else None


def _search(
    client: QdrantClient,
    encoded,
    company: str | None,
    top_k: int,
    collection: str,
    mode: str,
    chunk_kind: str | None = None,
):
    dense_vec, sparse_weights = encoded
    query_filter = _query_filter(company, chunk_kind)

    if mode == "dense":
        return client.query_points(
            collection_name=collection, query=dense_vec.tolist(), using=DENSE_VECTOR,
            query_filter=query_filter, limit=top_k,
        ).points
    if mode == "sparse":
        return client.query_points(
            collection_name=collection, query=to_sparse_vector(sparse_weights), using=SPARSE_VECTOR,
            query_filter=query_filter, limit=top_k,
        ).points

    # 각 브랜치가 top_k의 2배씩 후보를 내고 RRF로 합친다. 브랜치 limit이 top_k와 같으면
    # 한쪽에만 있는 정답이 융합 과정에서 밀려날 수 있다.
    #
    # ## RRF를 서버(FusionQuery)가 아니라 여기서 계산한다
    #
    # 서버 RRF는 **동점일 때 순서가 실행마다 다르다.** 같은 명령을 두 번 돌리면
    # `kb-capital-ratio`와 `naver-board`의 1·2위가 서로 뒤바뀌었다(README 3절).
    # 둘 다 RRF 0.5000 — dense 1위와 sparse 1위가 서로 다른 청크라 둘 다 1/(2+0)=0.5로
    # **정확히 같은 점수**이고, 서버는 동점 순서를 보장하지 않는다(병렬 세그먼트 병합).
    # 8문항에서 1위 하나가 바뀌면 recall@1이 12%p, 표 질문만 보면 25%p가 움직인다.
    #
    # 동점 규칙은 `fusion.rrf_fuse` 참고. 브랜치 안의 순서는 유사도가 실수라 정확한 동점이
    # 드물고, 흔들린 건 융합 단계뿐이었다.
    branch_limit = top_k * 2
    dense_hits = client.query_points(
        collection_name=collection, query=dense_vec.tolist(), using=DENSE_VECTOR,
        query_filter=query_filter, limit=branch_limit,
    ).points
    sparse_hits = client.query_points(
        collection_name=collection, query=to_sparse_vector(sparse_weights), using=SPARSE_VECTOR,
        query_filter=query_filter, limit=branch_limit,
    ).points
    return rrf_fuse(dense_hits, sparse_hits, top_k)


def rerank_hits(encoder, query: str, hits: list, top_k: int) -> list:
    if not hits:
        return hits
    scores = encoder.rerank(query, [h.payload.get("text", "") for h in hits])
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    reranked = []
    for i in order[:top_k]:
        hit = hits[i]
        # 원래 순위를 payload에 남겨둔다 — 리랭커가 무엇을 끌어올렸는지 --dump로 보려면 필요.
        hit.payload["_orig_rank"] = i + 1
        hit.score = scores[i]
        reranked.append(hit)
    return reranked


def main() -> None:
    ap = argparse.ArgumentParser(description="검색 품질 측정 하네스")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--kind", choices=("table", "narrative"), help="질문 종류로 한정")
    ap.add_argument("--dump", metavar="질문id", help="그 질문의 후보를 순위대로 출력")
    ap.add_argument("--dump-limit", type=int, default=15)
    ap.add_argument("--collection", default=COLLECTION_NAME)
    ap.add_argument("--mode", choices=("hybrid", "dense", "sparse"), default="hybrid", help="검색 방식 (기본 hybrid=RRF)")
    ap.add_argument(
        "--chunk-kind",
        choices=("p", "table"),
        help="이 종류의 청크만 검색한다 (p=서술, table=표). --kind(질문 종류)와 다르다",
    )
    ap.add_argument("--rerank", action="store_true", help="colbert(MaxSim)로 후보를 재정렬한다")
    ap.add_argument(
        "--rerank-depth",
        type=int,
        default=50,
        help="리랭킹할 후보 수 (기본 50). 이 개수만큼 검색한 뒤 재정렬해서 top-k로 자른다",
    )
    args = ap.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if args.dump:
        queries = [q for q in queries if q["id"] == args.dump]
        if not queries:
            raise SystemExit(f"질문 id '{args.dump}'를 찾을 수 없습니다.")
    elif args.kind:
        queries = [q for q in queries if q.get("kind") == args.kind]

    encoder = BgeM3Encoder(device=None)
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False, timeout=60)
    complete = _collection_report(client, args.collection, [q["company"] for q in queries if q.get("company")])
    print()

    rows = []
    for q in queries:
        encoded = encoder.encode_query(q["query"])
        # 리랭킹할 때는 후보를 depth만큼 넓게 가져와서 재정렬한 뒤 top_k로 자른다. 이래야
        # 리랭커가 "뒤에 있던 정답을 끌어올리는" 일을 할 수 있다.
        fetch_k = max(args.top_k, args.rerank_depth) if args.rerank else args.top_k
        hits = _search(client, encoded, q.get("company"), fetch_k, args.collection, args.mode, args.chunk_kind)
        if args.rerank:
            hits = rerank_hits(encoder, q["query"], hits, args.top_k)

        if args.dump:
            print(f"질문: {q['query']}  (회사 필터: {q.get('company') or '없음'})")
            print(f"라벨: {q['expect']}")
            # 먼저 라벨이 도구로 쓸 만한지 알려준다. 통과율이 높으면 아래 ★ 표시가
            # "정답을 찾았다"는 뜻이 아니라 "라벨이 아무나 통과시킨다"는 뜻이다.
            if _is_labeled(q["expect"]):
                density = _hit_density(hits, q["expect"])
                warn = "  <- 느슨하다. 순위 해석 불가" if density >= 0.4 else ""
                print(f"라벨통과율: 후보 {len(hits)}개 중 {density:.0%}가 ★{warn}")
            print("-" * 70)
            for rank, hit in enumerate(hits[: args.dump_limit], 1):
                p = hit.payload
                mark = "표" if _is_table(p) else "글"
                hit_mark = "★" if _matches(p, q["expect"]) else " "
                # 리랭킹했으면 원래 순위를 같이 보여준다 — 무엇이 끌어올려졌는지가 보인다.
                moved = f" (검색 {p['_orig_rank']}위)" if "_orig_rank" in p else ""
                print(f"{hit_mark}{rank:>3}. [{hit.score:.4f}][{mark}]{moved} {' > '.join(p.get('path') or [])}")
                print(f"      {p.get('text', '')[:110].replace(chr(10), ' ')}")
            return

        expect = q["expect"]
        rank = None
        if _is_labeled(expect):
            for i, hit in enumerate(hits, 1):
                if _matches(hit.payload, expect):
                    rank = i
                    break

        top10 = hits[:10]
        table_share = sum(1 for h in top10 if _is_table(h.payload)) / len(top10) if top10 else 0.0
        rows.append(
            {
                "id": q["id"],
                "kind": q.get("kind", "?"),
                "labeled": _is_labeled(expect),
                "rank": rank,
                # 결과 0건과 "점수 0"을 구분한다. 둘 다 0.0000으로 찍으면 회사가
                # 컬렉션에 없어 0건이 된 것을 "유사도가 0"으로 오해한다.
                "n_hits": len(hits),
                "top1": hits[0].score if hits else None,
                "table_share": table_share,
                # 라벨 느슨함. 이 값이 크면 그 질문의 recall/순위는 해석할 수 없다.
                "density": _hit_density(hits, expect) if _is_labeled(expect) else None,
            }
        )

    # 모드를 헤더에 박아둔다 — 출력 파일만 보고도 어떤 방식으로 잰 결과인지 알 수 있어야
    # 나중에 파일끼리 비교할 때 헷갈리지 않는다.
    # 리랭킹하면 점수가 MaxSim으로 바뀌므로 라벨도 바꾼다.
    score_label = "MaxSim" if args.rerank else ("RRF점수" if args.mode == "hybrid" else "top1점수")
    rerank_note = f" / rerank(depth={args.rerank_depth})" if args.rerank else ""
    # 청크 필터도 헤더에 박는다 — 빠뜨리면 kind=p 결과 파일이 전체 검색 결과와 구분되지 않는다.
    if args.chunk_kind:
        rerank_note += f" / 청크={args.chunk_kind}만"
    print(
        f"\n컬렉션 {args.collection} / mode={args.mode}{rerank_note}"
        f" / top-{args.top_k} / 질문 {len(rows)}개\n{'=' * TABLE_WIDTH}"
    )
    print(f"{'id':<24}{'종류':<10}{'첫정답순위':>10}{score_label:>10}{'top10표비율':>12}{'라벨통과율':>11}")
    print("-" * TABLE_WIDTH)
    for r in rows:
        if r["n_hits"] == 0:
            rank = "결과0건"
        elif not r["labeled"]:
            rank = "미라벨"
        else:
            rank = str(r["rank"]) if r["rank"] else f">{args.top_k}"
        top1 = f"{r['top1']:.4f}" if r["top1"] is not None else "-"
        # 라벨통과율이 높으면 그 줄의 순위는 신뢰할 수 없다. 눈에 띄게 표시한다.
        density = "-" if r["density"] is None else f"{r['density']:.0%}{'!' if r['density'] >= 0.4 else ''}"
        print(
            f"{r['id']:<24}{r['kind']:<10}{rank:>10}{top1:>10}{r['table_share']:>11.0%}{density:>11}"
        )

    labeled = [r for r in rows if r["labeled"]]
    if not labeled:
        print("\n라벨된 질문이 없어 지표를 계산하지 않았습니다.")
        return

    def summarize(subset: list[dict], name: str) -> None:
        if not subset:
            return
        n = len(subset)
        parts = [f"recall@{k} {sum(1 for r in subset if r['rank'] and r['rank'] <= k) / n:>5.0%}" for k in RECALL_AT if k <= args.top_k]
        mrr = sum(1 / r["rank"] for r in subset if r["rank"]) / n
        share = sum(r["table_share"] for r in subset) / n
        print(f"{name:<12}(n={n})  " + "  ".join(parts) + f"  MRR {mrr:.3f}  top10표비율 {share:.0%}")

    print("-" * TABLE_WIDTH)
    summarize(labeled, "전체")
    summarize([r for r in labeled if r["kind"] == "table"], "표 질문")
    summarize([r for r in labeled if r["kind"] == "narrative"], "서술 질문")

    # 라벨통과율이 높은 질문은 순위 변화를 해석할 수 없다 — 모드를 바꿔가며 비교하기 전에
    # 이걸 먼저 봐야 한다.
    loose = [r for r in labeled if (r["density"] or 0) >= 0.4]
    if loose:
        # f-string 안에서 같은 종류의 따옴표를 겹쳐 쓰면 3.11에서 SyntaxError다(3.12부터 허용).
        # 개발 환경이 섞여 있으므로 미리 문자열을 만들어 둔다.
        names = ", ".join(f"{r['id']}({r['density']:.0%})" for r in loose)
        print(
            f"\n!! 라벨통과율 40% 이상: {names}"
            "\n   후보 중 이만큼이 정답으로 계산된다 — 이 질문들의 recall/MRR 변화는 검색 품질이"
            "\n   아니라 라벨의 느슨함을 반영할 수 있다. --dump로 실제 청크를 보고 라벨을 좁힐 것."
        )

    if not complete or any(r["n_hits"] == 0 for r in rows):
        print(
            "\n!! 이 결과는 다른 컬렉션과 비교하면 안 된다 — 적재가 덜 끝나서 일부 질문이"
            " 결과 0건이다(위 경고 참고). 적재를 끝낸 뒤 다시 측정할 것."
        )
        return

    print(
        f"\n판정: 표 질문의 recall@{args.top_k}가 낮으면 청킹/어휘 문제(1~3단계 선행), "
        f"recall@{args.top_k}는 높은데 recall@1/5가 낮으면 순위 문제(리랭커 선행)."
    )


if __name__ == "__main__":
    main()
