# dart-filing-agent

DART(전자공시) 사업보고서를 대상으로 한 검색 파이프라인과 정형 값 조회 스토어입니다.
표가 많은 한국어 공시 문서에서 **서술 질문은 검색으로, 숫자 질문은 값 조회로** 답할 수 있게 만드는 것이 목적입니다.
LLM 답변 생성과 질문 라우팅(Agent 배선)은 이 저장소의 범위 밖이며, 여기서는 그 아래층인 검색과 조회를 만들고 측정했습니다.

- 대상: 5개사(삼성전자, KB금융, 삼성바이오로직스, SK이노베이션, NAVER)의 사업보고서 15건(2022~2025)
- 임베딩 모델: `BAAI/bge-m3` (로컬), 벡터DB: Qdrant([dart-filing-qdrant-server](https://github.com/SeonHwan/dart-filing-qdrant-server)에서 Docker로 기동)

## AI Agent 단계별 프로젝트

AI Agent를 단계별로 만들어 가는 5개 프로젝트 중 **2번째**입니다. 앞 프로젝트에서 남은 한계가 다음 프로젝트를 시작한 이유입니다.

| 순서 | 프로젝트 | 다룬 것 | 남은 한계 → 다음 단계 |
|---|---|---|---|
| 1 | [game-wiki-rag](https://github.com/SeonHwan/game-wiki-rag) | 가상 게임 위키 4문서로 RAG 기본 파이프라인, 하이브리드·리랭커·Contextual Retrieval·GraphRAG, tool-use 에이전트, MCP, LangGraph 기반 Adaptive/Self-RAG | 데이터와 평가 질문이 너무 작아(문서 4개, 질문 12개) 오케스트레이션, 답변 품질 개선, 여러 RAG 기법의 효과를 제대로 확인하기 어려움 → 실제 방대한 데이터로 다시 |
| 2 | **[dart-filing-agent](https://github.com/SeonHwan/dart-filing-agent)** | DART 사업보고서 5개사 15건(수만 개 청크): 표 행 단위 청킹, bge-m3 dense+sparse 하이브리드, 리랭커 측정, XBRL 값 조회 스토어 | 문서가 전문용어 중심이라 답변이 나아졌는지 확인하려면 용어부터 이해해야 해서 학습 진도가 더딤 → 일반 상식에 가까운 데이터로 |
| 3 | [encykorea-agent](https://github.com/SeonHwan/encykorea-agent) | 한국민족문화대백과사전 15,836건(청크 75,122개): hybrid 검색, 답변 생성 방식 비교, 직접 구현한 도구 호출 에이전트 | RAG 성격이 강해 에이전트 역량을 보여 주는 증거로 약하다고 판단 → 멀티 에이전트·오케스트레이션·LangGraph를 배우는 별도 프로젝트 |
| 4 | [logistics-agent](https://github.com/SeonHwan/logistics-agent) | 공개 물류 데이터(KTDB 화물 OD)와 정답이 계산되는 합성 시나리오: LangGraph 멀티 에이전트(supervisor, 병렬 분석가, 취합), 검증 노드, 사람 승인, Langfuse 관측, RAGAS 채점 | 병렬로 도는 에이전트를 더 늘려 멀티 에이전트의 효과를 눈으로 확인하고, MCP 분리와 서빙도 함께 다루려고 → 새 프로젝트로 |
| 5 | [kitchen-agent](https://github.com/SeonHwan/kitchen-agent) | 가상 주방 데이터와 6명의 전문 에이전트(LangGraph), 도구의 MCP 분리, RAGAS 평가, FastAPI 서빙 | — |

### 이전 단계([game-wiki-rag](https://github.com/SeonHwan/game-wiki-rag))에서 남은 한계
테스트 데이터셋이 너무 작아서 오케스트레이션, 답변 품질 개선, 여러 RAG 기법 적용을 효과적으로 해 보기 어려웠습니다.

### 이번 단계에서 바꾼 것
가상 문서 4개 대신 실제 공시인 DART 사업보고서 5개사 15건(수만 개 청크)으로 바꿨습니다. 표가 많은 문서에서 청킹과 검색을 다시 설계하고, 하이브리드 검색과 리랭커 적용 여부를 측정으로 정하고, 숫자 질문은 정형 값 조회로 풀었습니다. 다만 검색 평가 질문은 8개로 여전히 작았습니다(아래 '한계').

### 이 프로젝트의 한계가 다음 단계로 이어진 이유
공시 문서는 전문용어가 대부분이라, 답변이 개선됐는지 확인하려면 용어부터 이해해야 했고 그만큼 학습 진도가 더뎠습니다. 그래서 일반 상식에 가까운 개념이 많은 데이터로 바꿔 [encykorea-agent](https://github.com/SeonHwan/encykorea-agent)를 진행했습니다.

## 동기

사업보고서는 문서의 상당 부분이 표입니다(파싱한 표 39,307개). 문서를 문단처럼 고정 길이로 잘라 dense 임베딩으로 검색하면,
`"KB금융 자기자본비율"` 질문의 top-1 유사도가 0.39에 그치고 결과가 계정과목명도 기수도 없는 숫자 나열 조각이었습니다(개발 중 작업 기록에서 옮긴 수치이고, 원본 측정 파일은 저장소에 없습니다).
원인을 표 구조 손실과 숫자 표의 임베딩 특성으로 나누어 하나씩 고치고, 고칠 때마다 측정으로 효과를 확인했습니다.

## 구성

```
dart_xml_sanitize.py    well-formed가 아닌 DART XML 정제 (3종 결함)
dart_sections.py        목차 트리 추출 + 표 블록 추출 (헤더 판정, COLSPAN/ROWSPAN 격자 복원)
parse_filing.py         15건 → sections.json
chunk_sections.py       청킹 (문단은 그리디, 표는 행 단위 + 헤더 반복)
bge_m3.py               dense / sparse / colbert 인코더 래퍼
embed_and_index.py      인코딩 + Qdrant 적재 (named vector: dense + sparse)
fusion.py               RRF 융합 (동점 순서 고정)
eval_retrieval.py       검색 측정 하네스 (--mode dense|sparse|hybrid, --rerank, --dump)
fact_store.py           XBRL fact를 SQLite에 적재하고 조회 (find / series / query)
eval_facts.py           값 조회 평가 하네스
inspect_tables.py       원본 XML의 표 구조 측정 (읽기 전용)
view_filing.py          파싱·청킹 결과 HTML 뷰어
test_stage1.py          청킹·MaxSim·RRF 회귀 테스트 (모델 불필요)
test_fact_store.py      정형 스토어 회귀 테스트 (표준 라이브러리만)
```

## 핵심 설계와 근거

### 1. 표는 행 단위로 자르고 헤더를 모든 조각에 반복한다

표를 문자열 하나로 펴서 토큰 창으로 자르면 두 번째 조각부터 헤더가 없습니다. 표를 레코드 집합으로 보고 행을 자연 경계로 삼았습니다.

- 헤더 태그는 `TH` 하나뿐입니다. `TE`(368,954개)는 XBRL 수치 셀, `TU`(10,958개)는 값 셀이라, 이 둘을 헤더로 보면 재무제표가 5~11행 헤더를 가진 표로 오인됩니다.
- 헤더 행 판정은 "TH가 하나 이상 있고 TD가 없는 행"입니다. 부헤더 행이 TE와 TH를 섞어 쓰는 표가 있어서 "모든 셀이 TH"로는 놓칩니다.
- 병합 셀(COLSPAN/ROWSPAN)은 헤더뿐 아니라 본문에도 격자로 펼칩니다. ROWSPAN으로 셀이 빠진 행이 있는 표가 4,244개(다열 표의 약 18%)입니다.
- 1열 표(15,954개, 41%)는 표지 서식 같은 레이아웃이라 문단으로 되돌립니다.
- 캡션·단위는 검색용 `embedding_text`가 아니라 `text`에 넣습니다. 단위 없는 숫자 표는 검색돼도 답변에 쓸 수 없습니다.
- 캡션+헤더 고정부는 `MAX_TOKENS // 2`로 자릅니다. 없으면 열이 많은 표에서 8192토큰 청크가 만들어져 임베딩이 OOM 납니다.

### 2. dense만으로는 부족해서 sparse를 붙여 하이브리드로 쓴다

KB금융 청크 중 `자본총계`를 담은 것이 98개인데, dense 검색은 `"자본총계"` 질의로 top-50에 하나도 올리지 못했습니다(top-1 유사도 0.1603).
같은 질의를 sparse로 바꾸니 1위가 되었습니다. bge-m3의 lexical weight(sparse)는 `sentence-transformers`로 뽑을 수 없어서 `FlagEmbedding`을 썼고,
dense/sparse/colbert를 한 번의 forward로 얻습니다.

| 지표 (질문 8개, 41,436 청크, `eval_s3_*`·`eval_s5_*`) | dense | sparse | **hybrid** |
|---|---|---|---|
| 전체 recall@50 | 62% | 100% | **100%** |
| 전체 recall@10 | 38% | 75% | **88%** |
| 전체 recall@1 | 25% | 62% | **62%** |
| 전체 MRR | 0.286 | 0.677 | 0.671 |
| 표 질문 recall@10 | 50% | 75% | **100%** |

MRR은 sparse 단독이 근소하게 높지만 recall@10에서 hybrid가 앞서므로 hybrid를 택했습니다. dense가 이기는 질문도 있어서(`samsung-revenue-table` dense 5위 / sparse 11위) 한쪽에 몰지 않았습니다.
측정 파일: `eval_s3_dense.txt`, `eval_s3_sparse.txt`, `eval_s5_hybrid.txt`.

### 3. 서버 RRF의 동점 순서가 비결정적이어서 융합을 클라이언트로 옮겼다

같은 hybrid 측정을 두 번 돌리자 1위가 바뀌었습니다.

| 질문 | 1회차 | 2회차 |
|---|---|---|
| `kb-capital-ratio` | 1위 | 2위 |
| `naver-board` | 2위 | 1위 |
| 표 질문 recall@1 | 50% | 25% |

흔들린 질문은 전부 RRF 점수가 0.5000이었습니다. dense 1위와 sparse 1위가 서로 다른 청크이면 둘 다 `1/(2+0)`을 받아 정확히 동점이 되고(Qdrant RRF는 k=2, 순위는 0부터),
서버는 세그먼트를 병렬로 합치므로 동점 순서를 보장하지 않습니다. 두 실행의 `qdrant-client`·`FlagEmbedding`·`torch` 버전은 같았습니다(1.16.1 / 1.4.2 / 2.8.0).

`fusion.rrf_fuse`로 융합을 옮기고 동점을 "sparse 순위 → dense 순위 → id" 순으로 풀었습니다(이 데이터에서 sparse가 주력 신호라서). `qdrant-client`의 구현과 무작위 500회 대조해 점수가 모두 일치했고,
동점 규칙은 `test_stage1.py`가 고정합니다. 이후 같은 측정을 두 번 돌려 출력이 같은 것을 확인했습니다.

### 4. 리랭커(colbert MaxSim)는 기본 경로에서 뺐다

같은 모델이 colbert 벡터를 주므로 별도 리랭커 없이 top-50 후보를 재정렬해 볼 수 있습니다. 결정적 측정(`eval_s5_*`)에서 질문별 첫 정답 순위는 개선 1 / 악화 3 / 동일 4였고,
전체 MRR이 0.671 → 0.610으로 내려갔습니다. 이후 초미니 청크를 정리해 다시 적재한 컬렉션(36,449 청크, `eval_s9_*`)에서 다시 재면 개선 0 / 악화 1 / 동일 7, MRR 0.760 → 0.758이라 결정을 유지했습니다. 두 컬렉션은 청크 수가 달라 수치를 서로 비교하면 안 됩니다.
`--rerank` 플래그는 측정 도구로 남겼습니다.


**질의 EOS 토큰은 자르지 않습니다.** EOS가 문서마다 0.6~0.8로 고르게 높아 상수항처럼 보이지만, 모델이 EOS를 포함한 채 학습되었습니다
(FlagEmbedding 학습 코드의 `compute_colbert_score`는 EOS를 포함한 토큰 수로 나눕니다). EOS를 자른 조건으로 재측정하니 악화되었습니다.

| 지표 (rerank) | EOS 유지 | EOS 제거 |
|---|---|---|
| 전체 recall@10 | 75% | 62% |
| 전체 MRR | 0.610 | 0.553 |
| 표 recall@10 | 100% | 75% |
| `kb-capital-ratio` 순위 | 2위 | 24위 |

### 5. 숫자 질문은 청크 검색이 아니라 XBRL fact 조회로 푼다

값이 정답인 질문("KB금융 2025년 연결 자본총계")은 청크를 찾아도 숫자를 읽어야 합니다. 사업보고서의 숫자 셀(`TE`)은 이미 XBRL로 태깅되어 있어서 행·열 라벨을 파싱할 필요가 없었습니다.
`ACONTEXT`가 붙은 셀 62,717개 전부에 계정 코드(`ACODE`)가 있고, `ACONTEXT`는 기간(당기/전기/전전기), 연결/별도, 추가 차원을 담습니다.

- 값의 부호는 **표시 그대로**(괄호 = 음수)입니다. `ANEGATED`는 값 계산에 쓰지 않습니다. 같은 숫자가 보고서 연도에 따라 NEG=N/Y로 다르게 태깅되는 fact가 271건이고,
  현금흐름표의 "투자활동현금흐름 = 하위 항목 합"이 표시값으로는 60/60 성립하지만 ANEGATED를 반영하면 2025.12 보고서에서 30/30 실패합니다.
- `ADECIMAL`은 스케일이 아니라 정밀도이고, 원 단위 값은 `표시값 × 10^(-decimals)`입니다(`INF`·양수는 곱하지 않음).
- 같은 fact가 본표와 주석에 함께 있고 표시 단위가 다르면(천원 vs 원) 값이 천/백만 배 차이 납니다. 한 보고서 안 충돌 197건 중 157건이 이 경우여서, 본표 값을 우선합니다.

검증:

- 자산 = 부채 + 자본: 5개사 × 4개 연도 20/20 일치(`fact_check.txt`)
- 투자활동현금흐름 소계 = 항목 합: 60/60
- `eval_facts.py`: 값 질문 14문항 14/14. 정답은 정형 스토어를 거치지 않고 **원본 XML에서 직접 추출**했습니다(스토어의 버그가 정답과 답 양쪽에 들어가 채점이 순환하는 것을 막기 위해).
  `python eval_facts.py --verify-answers`가 정답 자체를 원문과 다시 대조합니다.
- 하네스가 틀린 규칙을 잡는지 확인하려고 규칙을 일부러 깨 보았습니다(옛 부호 규칙 12/14, 스케일 무시 4/14, INF에도 스케일 곱함 13/14).
  **본표 우선 규칙을 빼도 14/14라 이 평가로는 잡히지 않습니다.** 실데이터에서 그 규칙이 결과를 바꾸는 경우가 0건이기 때문이며, 그 규칙은 `test_fact_store.py`의 합성 케이스가 지킵니다.

## 한계

- 검색 평가셋은 **질문 8개**입니다. 질문 하나가 recall을 12~25%p 움직이므로 위 수치는 경향으로만 읽어야 합니다.
- 정답 라벨은 "질문의 핵심 용어가 본문에 있는 청크"입니다. 용어는 있지만 답이 아닌 청크도 정답으로 세어 recall이 과대평가될 수 있습니다.
  이를 재려고 후보 중 라벨을 통과하는 비율(라벨통과율)을 같이 출력합니다. `bio-cmo`는 100%(회사 전체가 CMO 사업이라 모든 청크가 통과)라 순위 변화를 해석할 수 없고,
  `kb-total-equity` 44%, `naver-board` 42%도 느슨합니다(`eval_s5_hybrid.txt`).
- `sk-risk`("사업 관련 위험 요소")는 정답이 50위 안에 오지 않습니다. 청킹 문제가 아니라 질의와 원문의 어휘 격차로 판단했습니다.
  질의의 "요소"가 "경쟁요소" 문단과 일치하고, 정답이 있는 위험관리 절은 "시장위험·신용위험·유동성위험"이라 "위험" 한 토큰만 겹칩니다.
  해결 후보는 질의 재작성과 목차 라우팅이며 구현하지 않았습니다(`eval_s9_risk_*.txt`).
- 표를 후보에서 빼는 실험에서 dense는 좋아졌지만(서술 recall@10 25% → 75%) hybrid에서는 이득이 없었습니다(MRR 0.557 → 0.544). 이 역시 서술 질문 3~4개 기준입니다.
- 정형 스토어는 2022~2025, 5개사 범위입니다. 파생 지표("부채비율", "ROE")는 계산하지 않았습니다.
- 정형 스토어 쪽 평가는 14문항입니다.

## 실행

환경변수는 `.env.example`을 복사해 `.env`로 만듭니다(`QDRANT_URL`). Qdrant는 `dart-filing-qdrant-server`에서 `docker compose up -d`로 띄웁니다.

```bash
pip install -r requirements.txt

python parse_filing.py                   # sections.json
python chunk_sections.py                 # chunks.jsonl
python embed_and_index.py                # 임베딩 + Qdrant 적재 (GPU/MPS 메모리가 부족하면 --batch-size 4, 그래도 안 되면 --device cpu)

python eval_retrieval.py --mode hybrid   # 검색 측정 (--mode dense|sparse, --rerank, --dump <질문id>)

python fact_store.py build               # XBRL fact 적재
python eval_facts.py --verify-answers    # 정답을 원문과 대조
python eval_facts.py                     # 값 조회 평가

python test_stage1.py                    # 청킹·MaxSim·RRF 테스트 (transformers와 bge-m3 토크나이저 필요)
python test_fact_store.py                # 정형 스토어 테스트
```
