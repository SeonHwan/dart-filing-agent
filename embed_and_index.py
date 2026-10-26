from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from bge_m3 import MAX_LENGTH, MODEL_NAME, BgeM3Encoder, release_accelerator_memory

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
COLLECTION_NAME = "dart_filings_hybrid"
EMBEDDING_MODEL = MODEL_NAME
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
CACHE_SUFFIX = "bge-m3-hybrid"
# 개발 환경이 통합 메모리 16GB인데, OOM이 난 시점에 `MPS allocated: 12.96 GiB`였다 — 16GB
# 머신에서 그 수치는 이미 스왑 중이라는 뜻이고, bge-m3를 올린 상태로 batch 32는 과하다. 배치를 줄여도 임베딩 결과는 사실상 같다(패딩 길이만
# 달라짐) — 품질이 아니라 메모리 문제라 배치가 맞는 손잡이다.
BATCH_SIZE = 8
UPSERT_BATCH_SIZE = 128

# 모듈 최상단에서 os.environ[...]으로 필수화하면 import만으로 KeyError가 난다
# (CI에서 import만으로 깨지지 않게). docstring과
# .env.example이 약속한 기본값을 코드에도 반영한다.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _point_id(rcept_no: str, index_in_filing: int) -> str:
    # atocid(목차 번호)는 문서 전체에서 고유하지 않다(KB금융 2025.12에서 서로 다른 섹션
    # 두 곳이 같은 ATOCID="4"를 가져 99개 청크가 충돌한다). chunks.jsonl 안에서의 등장 순서(항상 같은 순서로 재생성됨)를
    # 대신 키로 써서 고유성을 보장한다.
    key = f"{rcept_no}:{index_in_filing}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, tuple[np.ndarray, dict[int, float]]]:
    if not cache_path.exists():
        return {}
    with np.load(cache_path) as data:  # NpzFile은 파일 핸들을 잡고 있어서 닫아줘야 한다
        hashes = data["hashes"].tolist()
        dense = data["dense"]
        # sparse는 텍스트마다 길이가 달라서 npz에 그대로 못 담는다 -> CSR 형태로 눌러 담았다.
        indices, values, offsets = data["sparse_indices"], data["sparse_values"], data["sparse_offsets"]
    cache = {}
    for i, h in enumerate(hashes):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        cache[h] = (dense[i], dict(zip(indices[lo:hi].tolist(), values[lo:hi].tolist())))
    return cache


def _save_cache(cache_path: Path, cache: dict[str, tuple[np.ndarray, dict[int, float]]]) -> None:
    hashes = list(cache)
    indices: list[int] = []
    values: list[float] = []
    offsets: list[int] = [0]
    for h in hashes:
        weights = cache[h][1]
        indices.extend(weights)
        values.extend(weights.values())
        offsets.append(len(indices))
    np.savez(
        cache_path,
        hashes=np.array(hashes, dtype="<U64"),
        dense=np.array([cache[h][0] for h in hashes], dtype="float32"),
        sparse_indices=np.array(indices, dtype="int32"),
        sparse_values=np.array(values, dtype="float32"),
        sparse_offsets=np.array(offsets, dtype="int64"),
    )


def to_sparse_vector(weights: dict[int, float]) -> SparseVector:
    return SparseVector(indices=list(weights), values=list(weights.values()))


def ensure_collection(client: QdrantClient, dim: int, collection: str) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            # named vector여야 하이브리드 쿼리의 `using=`을 쓸 수 있다. 이름을 붙여두면
            # 같은 컬렉션에서 dense-only도 잴 수 있어 통제된 A/B가 된다(docstring 참고).
            vectors_config={DENSE_VECTOR: VectorParams(size=dim, distance=Distance.COSINE)},
            # bge-m3의 sparse는 **학습된** lexical weight다. BM25처럼 원시 빈도가 아니므로
            # Modifier.IDF를 붙이지 않는다 — 붙이면 이미 학습된 가중치에 IDF를 또 곱한다.
            sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams()},
        )
    # 인덱스가 없어도 payload 필터는 동작하지만 전수 스캔이다. Phase 3의 멀티문서 라우팅은
    # game-wiki-rag의 "컬렉션별 top_k 보장 병합"을 필터로 옮긴 구조라(collection 15개 대신
    # 단일 collection + payload 파티셔닝), 같은 쿼리를 필터만 바꿔 여러 번 던지게 된다 —
    # 그만큼 스캔 비용이 곱해진다. 같은 스키마로 다시 만드는 건 받아들여지므로 재실행 안전.
    # kind("p"/"table")로 표 청크만/서술 청크만 골라
    # 검색 품질을 따로 재고, 정형 스토어 단계에서 라우팅 조건으로도 쓴다.
    for field_name in ("company", "period", "rcept_no", "kind"):
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="청크를 bge-m3(dense+sparse)로 임베딩해서 Qdrant에 적재")
    # --collection이 필요한 이유: 이 스크립트는 필링 단위로 먼저 지우고 다시 넣으므로,
    # 청킹을 바꾼 뒤 그냥 재실행하면 비교 기준 컬렉션이 덮여서 사라진다.
    ap.add_argument("--collection", default=COLLECTION_NAME, help=f"적재할 컬렉션 (기본 {COLLECTION_NAME})")
    # 배치/디바이스를 코드 수정 없이 조절할 수 있게 빼둔다. 개발 환경이 통합 메모리 16GB
    # (팬 없음)라 메모리와 열 여유가 적다 — OOM이 나면 --batch-size 4, 그래도 안 되면
    # --device cpu 순서다(느리지만 안정적이고, 캐시가 있어서 재실행은 빠르다).
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"인코딩 배치 크기 (기본 {BATCH_SIZE})")
    ap.add_argument("--device", help="인코딩 디바이스 (예: cpu, mps). 생략하면 자동 선택")
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="fp16으로 인코딩(메모리 절약). dense 값이 미세하게 달라지므로 A/B 중에는 켜지 말 것",
    )
    args = ap.parse_args()
    collection = args.collection

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    encoder = BgeM3Encoder(device=args.device, use_fp16=args.fp16)
    # check_compatibility=False: PyPI의 qdrant-client 최신판(1.16.x)이 docker-compose가 받아온
    # qdrant/qdrant:latest(1.19.x)보다 뒤처져 있어서 뜨는 버전 경고를 무시함(REST API 기본
    # 동작인 upsert/count는 실제로 문제없이 동작하는 것 확인함).
    # timeout=60: 기본 타임아웃(5초)로는 청크 2천 개 넘는 파일 업서트 중 실제로
    # ReadTimeout이 나서 넉넉하게 늘림.
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False, timeout=60)

    total_points = 0
    for entry in manifest:
        filing_dir = DATA_DIR / entry["company"] / entry["period"].replace(".", "-")
        chunks = [json.loads(line) for line in (filing_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]

        texts = [c.get("embedding_text") or c["text"] for c in chunks]
        hashes = [_text_hash(t) for t in texts]

        lengths = encoder.token_lengths(texts)
        print(f"  실제 인코딩 토큰 길이: 최대 {max(lengths)}, 평균 {sum(lengths) / len(lengths):.0f}")
        if max(lengths) > MAX_LENGTH:
            worst = lengths.index(max(lengths))
            print(
                f"  !! 경고: {sum(1 for n in lengths if n > MAX_LENGTH)}개가 {MAX_LENGTH}토큰을 넘어"
                f" 잘립니다. 최대 {max(lengths)}토큰 (kind={chunks[worst].get('kind')})"
                f" @ {' > '.join(chunks[worst].get('path') or [])}"
            )

        cache_path = filing_dir / f"embeddings_cache.{CACHE_SUFFIX}.npz"
        cache = _load_cache(cache_path)
        # 청킹을 고쳐서 더 이상 존재하지 않는 청크의 벡터는 버린다(캐시 무한 증식 방지).
        current = set(hashes)
        cache = {h: v for h, v in cache.items() if h in current}

        # 같은 텍스트가 여러 번 나오면(표 캡션 등) 한 번만 인코딩한다.
        text_by_hash = dict(zip(hashes, texts))
        missing = [h for h in dict.fromkeys(hashes) if h not in cache]
        if missing:
            dense, sparse = encoder.encode(
                [text_by_hash[h] for h in missing], batch_size=args.batch_size
            )
            for h, d, s in zip(missing, dense, sparse):
                cache[h] = (d, s)
        _save_cache(cache_path, cache)  # 정리(prune)만 일어난 경우도 파일에 반영

        print(f"{entry['company']} {entry['period']}: 청크 {len(texts)}개, 새로 인코딩 {len(missing)}개(고유 텍스트 기준)")

        # 첫 필링에서만 컬렉션을 만든다 — dense 차원을 실제 벡터에서 읽어오려면 인코딩을
        # 한 번은 해봐야 한다(모델 설정을 신뢰하지 않고 실제 값을 쓴다).
        ensure_collection(client, len(cache[hashes[0]][0]), collection)

        # point id가 "rcept_no + 파일 내 등장 순번"이라, 청킹 로직을 바꿔 청크 수가 줄면
        # 뒤쪽 순번의 포인트가 그대로 고아로 남는다(upsert는 덮어쓰기만 하고 사라진 id를
        # 지우지는 않는다). 필링 단위로 먼저 비우고 다시 넣어서 컬렉션이 항상 chunks.jsonl과
        # 일치하게 만든다. 원자적이지 않으므로 이 사이에 죽으면 그 필링만 비어 있게 되는데,
        # 재실행하면 복구된다.
        client.delete(
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="rcept_no", match=MatchValue(value=entry["rcept_no"]))])
            ),
        )

        points = []
        for i, (chunk, h) in enumerate(zip(chunks, hashes)):
            dense_vec, sparse_weights = cache[h]
            points.append(
                PointStruct(
                    id=_point_id(chunk["rcept_no"], i),
                    vector={
                        DENSE_VECTOR: dense_vec.tolist(),
                        SPARSE_VECTOR: to_sparse_vector(sparse_weights),
                    },
                    payload=chunk,
                )
            )
        # 한 번에 다 보내면 REST 페이로드 한도(기본 32MB)를 넘는 건이 있어서(KB금융 등,
        # 청크 2천 개 이상) UPSERT_BATCH_SIZE 단위로 나눠 보낸다.
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            client.upsert(collection_name=collection, points=points[i : i + UPSERT_BATCH_SIZE])
        total_points += len(points)
        print(f"{entry['company']} {entry['period']}: {len(points)}개 벡터 적재")
        # 다음 필링을 인코딩하기 전에 가속기 캐시를 비운다 — OOM이 첫 필링을 끝낸 직후
        # (12.96GiB 점유 상태) 두 번째 필링에서 났기 때문이다.
        release_accelerator_memory()

    count = client.count(collection).count
    print(f"완료. 이번 실행 {total_points}개 업서트, 컬렉션 총 {count}개 포인트")


if __name__ == "__main__":
    main()
