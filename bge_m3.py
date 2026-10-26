from __future__ import annotations

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
# 어텐션 마스크가 batch x L x L로 잡히므로 L을 묶어야 한다. 청킹이 512 예산으로 자르고
# 표 청크의 최악이 1.5배(768)라 1024면 넉넉하다. bge-m3의 기본값 8192로 두면 batch 32에서
# 마스크만 8GiB이고, 실제로 MPS에서 그 크기 할당을 시도하다 OOM이 났다.
MAX_LENGTH = 1024


def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return 0.0
    sim = query_vecs @ doc_vecs.T  # [질의토큰, 문서토큰]
    return float(sim.max(axis=1).mean())


class BgeM3Encoder:

    def __init__(self, device: str | None = None, use_fp16: bool = False) -> None:
        # FlagEmbedding은 무거워서 지연 import한다 — 이 모듈을 상수만 쓰려고 import하는 쪽
        # (컬렉션 이름 등)이 모델 로딩 비용을 물지 않게.
        from FlagEmbedding import BGEM3FlagModel

        kwargs = {"use_fp16": use_fp16}
        if device:
            kwargs["devices"] = device
        self.model = BGEM3FlagModel(MODEL_NAME, **kwargs)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(ids) for ids in self.tokenizer(texts, add_special_tokens=True)["input_ids"]]

    def encode(
        self, texts: list[str], batch_size: int = 8
    ) -> tuple[np.ndarray, list[dict[int, float]]]:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype="float32")
        sparse = [
            {int(k): float(v) for k, v in weights.items() if float(v) > 0.0}
            for weights in out["lexical_weights"]
        ]
        return dense, sparse

    def encode_query(self, text: str):
        dense, sparse = self.encode([text], batch_size=1)
        return dense[0], sparse[0]

    def colbert_vecs(self, texts: list[str], batch_size: int = 8) -> list[np.ndarray]:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=MAX_LENGTH,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        return [np.asarray(v, dtype="float32") for v in out["colbert_vecs"]]

    def rerank(self, query: str, texts: list[str], batch_size: int = 8) -> list[float]:
        if not texts:
            return []
        query_vecs = self.colbert_vecs([query], batch_size=1)[0]
        doc_vecs = self.colbert_vecs(texts, batch_size=batch_size)
        return [maxsim(query_vecs, doc) for doc in doc_vecs]


def release_accelerator_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
