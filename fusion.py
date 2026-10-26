from __future__ import annotations

# Qdrant 서버 RRF와 같은 상수(qdrant_client/hybrid/fusion.py의 DEFAULT_RANKING_CONSTANT_K).
# 점수 = 1 / (k + 브랜치 내 순위), 순위는 0부터. k=2라 dense 1위만 = 1/2, sparse 1위만 = 1/2 —
# 양쪽 1위가 서로 다른 청크면 **정확히 동점**이다. 이 값을 바꾸면 이전 측정과 비교할 수 없다.
RRF_K = 2


def rrf_fuse(dense_hits: list, sparse_hits: list, top_k: int) -> list:
    dense_rank = {h.id: i for i, h in enumerate(dense_hits)}
    sparse_rank = {h.id: i for i, h in enumerate(sparse_hits)}
    # 한쪽 브랜치에 없는 hit의 순위. 어떤 실제 순위보다 뒤여야 동점 규칙이 "있는 쪽 우선"이 된다.
    absent = len(dense_hits) + len(sparse_hits)
    by_id = {h.id: h for h in dense_hits}
    by_id.update({h.id: h for h in sparse_hits})

    def score(pid) -> float:
        return sum(1 / (RRF_K + ranks[pid]) for ranks in (dense_rank, sparse_rank) if pid in ranks)

    # id는 UUID 문자열이라 str()로 비교하면 실행과 무관하게 같은 순서가 나온다.
    order = sorted(
        by_id,
        key=lambda pid: (-score(pid), sparse_rank.get(pid, absent), dense_rank.get(pid, absent), str(pid)),
    )
    fused = []
    for pid in order[:top_k]:
        hit = by_id[pid]
        hit.score = score(pid)
        fused.append(hit)
    return fused
