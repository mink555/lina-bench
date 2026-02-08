# AI TMR Assistant — Multi-Turn Tool-Calling Benchmark

폐쇄망 MCP 구축을 위한 오픈소스 LLM 선정 벤치마크.
5개 모델의 Agent 역량(tool calling 정확도)과 한국어 답변 생성을 106턴 멀티턴 스트레스 테스트로 비교한다.

---

## 1. 배경 및 목적

### 1.1 폐쇄망 MCP 구축과 오픈소스 LLM 선정

폐쇄망 환경에서 MCP(Model Context Protocol)를 구축하기 위해 자체 호스팅 가능한 오픈소스 LLM을 선정해야 한다.
모델 "성능"을 2가지 축으로 정의하고 비교한다.

| 축 | 측정 내용 | 세부 지표 | 비중 |
|---|---|---|---|
| 1. Agent 역량 | tool calling 정확도 | 호출 정확도(단일/병렬), 인자 정확도, 행동 판단(호출/미호출) | 핵심 |
| 2. 한국어 답변 생성 | 자연어 응답 품질 | 관련성·정확성·간결성·적절성 | 서브 |

Agent 역량을 중점으로 보는 이유 — MCP의 핵심 가치는 LLM이 외부 도구를 정확히 호출하는 것이다.
tool을 잘못 호출하거나 인자를 틀리면 후속 파이프라인 전체가 실패하므로, 답변 품질보다 tool calling 정확도가 우선한다.

| 핵심 지표 | 측정 내용 | 업계 대응 |
|-----------|----------|----------|
| Tool Acc | 올바른 tool 선택 (단일/병렬) | Function name accuracy |
| Arg Acc | 인자 정확도 | Parameter extraction accuracy |
| No-Call | 불필요 호출 억제 | Abstention / Irrelevance detection |
| FC Judge | 행동 판단 정확도 | AST-based behavioral correctness |

한국어 답변 생성(NL Quality)은 현재 서브지표이나, 실제 서비스로 이어질 때 고객 응대 품질로 직결되므로 함께 관리한다.

### 1.2 AI TMR Assistant 챗봇 PoC

MCP 구축과 동시에 AI TMR(텔레마케터) Assistant 챗봇 PoC를 수행해야 한다.
TMR이 고객과 영업콜을 하면서 실시간으로 챗봇에 질문하는 상황을 가정한다.

이때 핵심 질문은:

> "몇 turn까지 모델이 정확하게 tool을 호출할 수 있는가?"

실무 TMR 콜은 보통 5~7턴이지만, 복잡한 상담은 그 이상으로 이어질 수 있다.
모델이 몇 턴에서 성능이 붕괴(Critical Threshold 85% 이하)되는지 파악하면,
PoC 개발 시 허용 가능한 턴 수와 아키텍처 설계의 근거를 확보할 수 있다.

| 검증 역할 | 질문 | 지표 |
|-----------|------|------|
| Agent (도구 실행 + 행동 판단) | 올바른 tool? 정확한 인자? 동시 호출? 호출/미호출 판단? | Tool Acc, Arg Acc, FC Judge |
| Chatbot (답변 생성) | 자연스럽고 적절한 답변인가? | NL Quality |

따라서 본 벤치마크는 AI TMR 시나리오를 직접 설계하여 5개 오픈소스 모델을 비교·검증하고,
모델 선정 근거 + 실무 허용 턴 수 + 성능 개선 방향을 도출하는 것을 목표로 한다.

---

## 2. 벤치마크 프레임워크 선정

### 2.1 Tool-Calling 벤치마크 3종 리서치

tool calling 역량을 검증하는 주요 벤치마크 3개를 리서치하였다.

| | [BFCL v4][bfcl] | [FunctionChat][functionchat] | [τ²-bench][tau-bench] |
|---|---|---|---|
| 개발 | UC Berkeley | Kakao | Sierra Research |
| 언어 | 영어 | 한국어 | 영어 |
| 측정 초점 | Tool 호출 정밀도 | 상황별 행동 판단 | E2E Task 완수율 |
| 평가 방식 | AST 매칭 (구조적 비교) | [LLM-as-Judge][llm-judge] | Pass rate (전 과정 실행) |
| Multi-turn | O (v4부터) | O | O |
| Parallel Call | O | — | — |
| No-Call 판단 | △ (일부) | O (Slot Q + Relevance) | — (환경 내장) |
| 환경 구축 | 불필요 (offline) | 불필요 (offline) | 필요 (tool 백엔드 + user sim) |

### 2.2 선정: BFCL v4 + FunctionChat

2개를 선정한 이유 — 하나만 쓰면 빠지는 평가 차원이 있다:

- BFCL v4만 사용 시: No-Call 세분화(Slot Question, Relevance Detection) 없음, 한국어 답변 품질 평가 없음
- FunctionChat만 사용 시: args 수준의 AST 정밀도 없음, parallel call 평가 빠짐

두 프레임워크를 상호 보완적으로 결합하여 Agent 역량의 전체 차원을 커버한다.

| | BFCL v4에서 차용 | FunctionChat에서 차용 |
|---|---|---|
| 평가 방식 | AST 매칭 (tool name + args 구조적 비교) | [LLM-as-Judge][llm-judge] (GPT-4o) |
| 핵심 역할 | tool name/args 정밀 채점, parallel 감지 | 미호출 판단, 추가질문 평가, 답변 품질 |
| 지표 | Tool Acc, Arg Acc | FC Judge, NL Quality |

### 2.3 τ²-bench 제외 사유

[τ²-bench][tau-bench]는 에이전트와 사용자 모두가 공유 환경을 수정하는 Dual-Control E2E 벤치마크이다.

- 환경 구축 부담: tool 실행 백엔드(DB + API)와 user simulator를 도메인별로 구축해야 함
- 도메인 불일치: 지원 도메인(airline, retail, telecom)에 보험 TMR이 없어 도메인 재설계 필요
- 현 단계 목표와 불일치: PoC 단계에서는 모델의 raw function calling 정확도를 먼저 검증하는 것이 우선. E2E 환경 구축은 ROI가 낮음

---

## 3. 시나리오 설계

### 3.1 설계 목적

TMR이 고객 통화 중 AI 챗봇에 입력하는 Decision Point를 시뮬레이션한다.

- Turn = TMR이 AI 챗봇에 1회 입력 (고객 대화 턴 ≠ 챗봇 턴)
- TMR은 판단이 필요한 순간에만 챗봇 사용
- TMR이 아는 것: 고객 연령·성별·가입 상품 (통화 전 기본 정보)

### 3.2 콜 유형 — 도메인 반영

TMR 영업콜을 결과 기준으로 3가지 유형으로 분류하였다.
각 유형은 대화 길이와 tool 사용 패턴이 다르므로, 시나리오 설계 시 구분이 필요하다.

| 콜 유형 | 설명 | 실무 턴 수 | 스트레스 턴 |
|---------|------|-----------|------------|
| O1 청약 | 고객이 가입 의사 → 조건 확인·보험료 산출·청약 절차 | ~7턴 | 18~19턴 |
| O2 보류 | 고객이 고민/비교 → 추천·비교 후 재연락 약속 | ~5턴 | 16~17턴 |
| ~~O3 실패~~ | 고객 거절 → 즉시 종료 | 2~4턴 | 제외 |

실무 턴 수는 공개 데이터셋이 아닌, 보험 TM 표준 콜 스크립트 구조에서 도출한 추정치이다.
TMR이 챗봇에 판단을 요청하는 Decision Point 기준으로, 콜 스크립트의 단계 수와 대응한다:

| 유형 | 콜 스크립트 단계 (Decision Point) | 턴 수 |
|------|-----------------------------------|-------|
| 청약 | 본인확인 → 니즈파악 → 건강고지 → 상품추천 → 보험료산출 → 특약설명 → 청약안내 | ~7 |
| 보류 | 본인확인 → 니즈파악 → 상품추천 → 비교설명 → 재연락약속 | ~5 |
| 실패 | 본인확인 → 니즈파악 → 재연락약속 → 종료 | 2~4 |

실패콜(O3) 제외 이유: 2~4턴으로 빠르게 종료되므로 스트레스 테스트에 필요한 턴 수(15~19턴) 확보가 불가능하다.

### 3.3 Multi-Turn Stress Test — 3축 설계

실무 대비 2.5~3배 턴을 늘리되, "어떻게 어렵게 만드느냐"를 3축으로 분리한다.
순수 모델의 붕괴를 파악하고 원인을 분석하려면, 스트레스 원인을 독립적으로 분리해야 하기 때문이다.

설계 근거 — 인지심리학의 기억 실패 3대 원인 (MECE, [Wixted 2004][forgetting]):
LLM도 동일한 패턴을 보인다는 점이 [Lost in the Middle][lost-middle] 등에서 실증되었다.

| 인지적 원인 | 스트레스 축 | 코드 | 설명 | 테스트 포인트 |
|------------|-----------|------|------|-------------|
| Information Overload | 조건누적 | ST1 | 매 턴마다 새 조건 추가 (고혈압→흡연→갑상선→당뇨) | 누적된 전체 state 유지 |
| Decay over Distance | 맥락희석 | ST2 | 다양한 토픽이 분산 (입원→특약→Q&A→비교→정리→업셀) | 초반 context 희석에 대한 저항 |
| Interference | 교란주입 | ST3 | 고객이 번복·전환 (흡연→비흡연, 암→건강→암, 6만→3만→5만) | state 갱신 (이전 값을 올바르게 수정) |

세 원인은 상호 독립적이며, 멀티턴 대화에서 모델이 실패할 수 있는 주요 차원을 빠짐없이 커버한다.

3축 분리 원칙: ST1은 조건만 쌓임, ST2는 토픽만 분산, ST3은 번복/전환만 (조건 누적 없음).
교차분석은 "어떤 스트레스가 더 치명적인가"의 방향성 분석에 활용한다.

### 3.4 시나리오 매트릭스 및 상세

매트릭스 (2×3 = 6개)

| | ST1 조건누적 | ST2 맥락희석 | ST3 교란주입 |
|---|---|---|---|
| O1 청약 | O1_ST1 (19턴) | O1_ST2 (18턴) | O1_ST3 (19턴) |
| O2 보류 | O2_ST1 (17턴) | O2_ST2 (16턴) | O2_ST3 (17턴) |

총 106턴 (single 82 + parallel 12 + no_call 12), 20개 tool 전부 사용.

<details>
<summary>시나리오 상세</summary>

| 시나리오 | 고객 (보유 → 상담) | 주요 교란/스트레스 | 핵심 테스트 |
|----------|-------------------|-------------------|------------|
| O1_ST1 | 45세 남성, 건강보험 보유 → 암보험 상담 | 고혈압→흡연→갑상선→당뇨 | T19에서 4개 조건 모두 기억하는가 |
| O1_ST2 | 38세 여성, 실손보험 보유 → 입원보장 상담 | 특약→Q&A→비교→정리→예산→전환→업셀 | 토픽 전환 후 초반 맥락 유지 |
| O1_ST3 | 42세 남성, 종신보험 보유 → 암보험 상담 | 흡연→비흡연, 암→건강→암, 6만→3만→5만 | state 번복 후 올바른 값 갱신 |
| O2_ST1 | 50세 남성, 암보험 보유 → 건강보험 상담 | 당뇨→간→혈압 | 3개 조건 누적 기억 |
| O2_ST2 | 35세 여성, 실손보험 보유 → 암보험 상담 | 비교→치아→추천→업셀 | 폭넓은 토픽에서 맥락 유지 |
| O2_ST3 | 47세 남성, 건강보험 보유 → 종신보험 상담 | 종신→간병→종신, 10만→4만→7만 | 상품·예산 이중 번복 처리 |

</details>

### 3.5 당사 특화

| 특화 요소 | 시나리오 반영 |
|-----------|-------------|
| 치아보험 (충치·임플란트·보철) | O2_ST2에서 임플란트 이력 고객 |
| 간병보험 (뇌질환·장기요양) | O2_ST3에서 간병 전환 |
| 55세 간편보험 | O2_ST1에서 유병자 간편심사 |

### 3.6 Parallel / No-Call Turn 설계

Parallel / Multiple Function Call
— [BFCL v4][bfcl]의 Parallel(다른 tool 동시) + Parallel-Multiple(같은 tool, 다른 args)을 반영.
시나리오당 2개 × 6 = 12개 parallel turn. Single과 별도 스코어 산출.

No-Call 턴: Slot Question + Relevance Detection
— [FunctionChat][functionchat]의 Slot Question(추가 질문) + Relevance Detection(범위 밖 거부)을 반영.
시나리오당 2개 × 6 = 12개 no_call turn.

- Slot Question: 필수 정보 누락 시 tool 대신 질문 → SLOT-all 평가 (누락된 정보를 전부 물어봤는가?)
- Relevance Detection: 보험 외 요청(적금·대출 등)을 tool 없이 거절

### 3.7 Turn-Point 평가

1개의 긴 시나리오를 T3·T5·T7·T10·T13·T15·T17·T19 지점에서 슬라이싱하여 성능 곡선을 생성.
"몇 턴까지 85%를 유지하는가?"를 바로 확인 가능.

---

## 4. Tool 설계서 (20개)

### 4.1 설계 원칙

- MECE: 4개 카테고리 20개 tool, 기능 중복 없음
- 당사 특화: 치아(충치·임플란트·보철), 간편보험, 간병보험
- TMR 현실 반영: CRM 없음, 본인 보유 상품만 조회, 개인정보 위반 차단

### 4.2 Function 분류

| 카테고리 | 수 | 대표 tool | 역할 |
|----------|---|-----------|------|
| F1 정보조회 | 10 | product_lookup, coverage_detail_lookup, underwriting_rules_lookup | 상품·보장·인수 정보 |
| F2 판단/계산 | 4 | underwriting_eligibility_checker, premium_calculator | 가입 가능성·보험료 |
| F3 추천 | 4 | product_recommender, budget_optimizer | 니즈/예산 기반 추천 |
| F4 규제 | 2 | compliance_checker, privacy_violation_detector | 규정·개인정보 |

### 4.3 개인정보 보호

| 허용 | 차단 |
|------|------|
| 본인 보유 보험, 거절 이력 | 콜·문의·납입·청구 이력, 의료 기록, 타인 정보 |

---

## 5. 평가 지표

평가 지표는 Agent 역량(핵심)과 답변 생성(서브) 2축으로 설계한다.
가중치 없이 개별 산출하고 교차 분석한다.

### 5.1 지표 요약

| 축 | 지표 | 출처 | 집계 범위 | 측정 내용 |
|---|------|------|----------|----------|
| Agent (핵심) | Tool Acc | BFCL v4 | tool_call 턴만 | 정답 tool 선택 + 인자 정확도 |
| Agent (핵심) | FC Judge | FunctionChat | 전체 턴 | 행동 판단 (호출/미호출/개인정보) |
| 답변 (서브) | NL Quality | LLM-as-Judge | 텍스트 있는 턴 | 자연어 응답 품질 (GPT-4o) |

Agent 역량이 핵심인 이유: tool을 잘못 호출하면 후속 파이프라인 전체가 실패한다.
NL Quality는 현재 목적에서 서브지표이나, 실제 서비스로 이어질 때 고객 응대 품질로 직결되므로 함께 관리한다.

### 5.2 Agent 역량: Tool Acc — BFCL Score (tool_call 턴 전용)

| Sub-metric | 설명 | 비고 |
|------------|------|------|
| tool_name_acc | 정답 tool 이름 exact match | |
| arg_key_acc | GT key 존재 비율 | tool name 정답일 때만 산출 |
| arg_value_acc | key-value deep compare | tool name 정답일 때만 산출 |

- tool name이 틀리면 arg = 0 ([BFCL v4 원본 방식][bfcl-paper])
- Parallel: greedy 매칭 후 개별 평가, `parallel_detected` = 복수 호출 인식
- no_call 턴은 BFCL 집계에서 완전 제외 — 미호출 판단은 FC Judge가 전담
- Multi-turn Memory: 후반 GT args에 앞 턴 조건이 모두 포함, 하나라도 잊으면 arg 하락

### 5.3 Agent 역량: FC Judge — 행동 판단 (전체 턴)

| Sub-metric | Tool Call 턴 | No-Call 턴 |
|------------|-------------|-----------|
| action_type_acc | 호출=1, 미호출=0 | 미호출=1, 호출=0 |
| tool_selection_acc | 이름 일치=1 | 미호출=1, 호출=0 |
| privacy_detection_acc | privacy 턴 정답=1 | 항상 1.0 |

### 5.4 답변 생성: NL Quality (서브)

GPT-4o가 관련성·정확성·간결성·적절성을 Pass/Fail 판정 ([LLM-as-Judge][llm-judge] 방식). 텍스트 없으면 스킵.

현 단계에서는 모델 간 비교를 위한 참고 지표로 활용한다.
향후 실제 서비스 구축 시, 고객 응대 품질의 핵심 지표로 승격될 수 있다.

### 5.5 Performance — 종합 점수

```
tool_call 턴:  (Tool Acc + Arg Acc + FC Judge) / 3
no_call 턴:    FC Judge만
전체 Performance = 모든 턴의 per-turn 점수 평균 (micro-average)
```

### 5.6 지표 무결성 설계

BFCL(Tool/Arg)과 FC Judge의 집계 범위가 다르므로, 교차 오염을 방지하는 구조:

```
                     ┌─ tool_call 턴 ─┐   ┌─ no_call 턴 ─┐
  Tool Acc (BFCL)    │    산출         │   │   제외        │
  Arg Acc  (BFCL)    │    산출         │   │   제외        │
  FC Judge           │    산출         │   │   산출        │
  NL Quality         │    산출         │   │   산출        │
  Performance        │  (T+A+FC)/3    │   │   FC만        │
                     └────────────────┘   └──────────────┘
```

- `evaluate_bfcl_no_call()` → 항상 0.0 반환 (마커 역할)
- 집계·Turn-Point·리포트 모든 경로에서 `exclude_no_call=True` 적용
- NC:Acc = FC Judge의 `action_type_acc` 기준 (BFCL 아님)

---

## 6. 실험 환경

### 6.1 대상 모델

| 모델 | 개발사 | 파라미터 | 아키텍처 | FP16 VRAM (추정) |
|------|-------|---------|---------|-----------------|
| llama-3.3-70b-instruct | Meta | 70B | Dense | ~140GB |
| mistral-small-3.2-24b-instruct | Mistral | 24B | Dense | ~48GB |
| qwen3-32b | Alibaba | 32B | Dense | ~64GB |
| qwen3-14b | Alibaba | 14B | Dense | ~28GB |
| qwen3-next-80b-a3b-instruct | Alibaba | 80B (3B active) | MoE | ~160GB |

선정 기준: tool calling 지원, 한국어 성능, 오픈소스, 다양한 파라미터 규모(14B~80B).

### 6.2 환경 제약

- GPU: H100 80GB × 1대 (폐쇄망)
- 제한: 서빙과 학습을 동시에 수행할 수 없음 → 파인튜닝 배제
- 배포 조건: FP16 서빙이 가능한 모델이 유리 (양자화 시 성능 저하 리스크)
- 벤치마크 실행: OpenRouter API를 통해 원본 모델의 순수 성능을 측정 (양자화 영향 배제)

qwen3-14b(~28GB)는 H100 80GB에서 FP16 서빙 시 52GB 여유 → 배치 처리·동시 요청 대응에 유리.

### 6.3 프로젝트 구조

```
my_bench/
├── README.md
├── .env                          ← OPENROUTER_API_KEY
├── requirements.txt
├── scenarios/
│   ├── scenarios_6_multi_turn.jsonl  ← 6개 시나리오 (106턴) — GT 데이터
│   ├── create_sc.py                  ← 시나리오 설계 시각화 (stress_test_summary.png)
│   └── stress_test_summary.png       ← 3축 스트레스 설계 차트
├── tool_specs/
│   └── tools_spec.py             ← 20개 tool 명세서
├── benchmark/
│   ├── evaluator.py              ← 평가 모듈 (BFCL AST + FC Judgment + LLM Judge)
│   ├── run_benchmark.py          ← 벤치마크 실행기 + Turn-Point 분석
│   ├── compare_results.py        ← 결과 비교 + 텍스트 리포트
│   └── results/
└── experiments/                  ← 성능 개선 실험
```

### 6.4 실행 방법

```bash
# 전체 실행
python -m benchmark.run_benchmark

# Dry-run (API 호출 없이 코드 검증)
python -m benchmark.run_benchmark --dry-run

# 결과 비교 + 리포트
python -m benchmark.compare_results
```

산출물: 텍스트 리포트 (`benchmark/results/report_*.txt`)

---

## 7. 실험 결과 및 분석

106턴 × 5모델 | Judge: GPT-4o | 실무 기준: ~T7 누적

### 7.1 종합 성적표

| 모델 | Tool | Arg | FC | NL | 실무 Perf | 전체 Perf |
|------|------|-----|----|----|-----------|-----------|
| llama-3.3-70b-instruct | 85.6% | 36.5% | 84.4% | N/A | 68% | 67% |
| mistral-small-3.2-24b-ins | 42.6% | 31.1% | 66.7% | 28% | 44% | 51% |
| qwen3-32b | 77.1% | 56.3% | 85.1% | 43% | 76% | 72% |
| **qwen3-14b** | **83.0%** | **59.2%** | **87.4%** | **50%** | **82%** | **75%** |
| qwen3-next-80b-a3b-instru | 78.2% | 58.5% | 86.9% | 24% | 77% | 76% |

### 7.2 모델 선정: qwen3-14b

**1위: qwen3-14b** — 실무 Performance **82%**, Agent + 답변 생성 겸용 가능.

| 선정 근거 | 내용 |
|----------|------|
| Agent 1위 | 실무 82% (Tool 93%, Arg 80%, FC 96% @T7) |
| NL 1위 | 답변 품질 50% (5개 모델 중 최고) → 1모델로 Agent+답변 겸용 |
| H100 적합 | 14B ~28GB FP16 → 80GB H100에서 52GB 여유, 배치/동시 요청 대응 가능 |
| 양자화 불필요 | FP16 서빙 가능 → 양자화로 인한 성능 저하 리스크 없음 |

참고: 전체 Perf 1위는 qwen3-next-80b(76%)이나, 실무 구간에서 qwen3-14b(82%)가 5%p 앞선다.
또한 80B MoE는 FP16 ~160GB로 H100 1대에 탑재 불가 (양자화 필수).

**1위 모델** 실무 구간(@T7) 세부 진단 — 42턴 (tool_call 36 + no_call 6):

| 지표 | 점수 | 상태 |
|------|------|------|
| Tool Acc | 93% | 우수 |
| Arg Acc | 80% | 병목 |
| FC Judge | 96% | 우수 |
| No-Call | 33% | 병목 |
| Performance | 82% | |

개선 우선순위: No-Call(33%) > Arg Acc(80%) > Tool Acc(93%)

### 7.3 Turn-Point 분석 — 붕괴 지점

시나리오를 T3~T19 지점에서 슬라이싱하여 누적 Performance를 계산한다.
(90%+ SAFE / 85%+ GOOD / 75%+ RISK / <75% DANGER)

| 모델 | ~T3 | ~T5 | ~T7 | ~T10 | ~T13 | ~T15 | ~T19 |
|------|-----|-----|-----|------|------|------|------|
| llama-3.3-70b | 79% | 67% | 68% | 70% | 65% | 65% | 67% |
| mistral-small-3.2-24b | 31% | 36% | 44% | 44% | 48% | 49% | 51% |
| qwen3-32b | 85% | 73% | 76% | 75% | 71% | 72% | 72% |
| **qwen3-14b** | **88%** | **81%** | **82%** | **79%** | **77%** | **75%** | **75%** |
| qwen3-next-80b | 79% | 78% | 77% | 75% | 75% | 75% | 76% |

주요 발견:

- qwen3-14b는 ~T3까지 85%+ 유지, 이후 점진적 하락. 실무 7턴 구간에서 82%로 가장 높음
- 붕괴 순서: 인자(Arg) → 도구(Tool) → 행동(FC) (4/5 모델 공통)
  — 턴이 늘어남에 따라 먼저 인자를 잊고, 그다음 잘못된 tool을 호출하고, 마지막으로 행동 판단이 무너짐
- 실무 권장: 7턴 이내 사용 시 82%, 충분히 활용 가능. T7 이후는 내구도 진단용이며 운영 목표가 아님

### 7.4 능력 해부 — Single / Parallel / No-Call

Single (82턴) — tool 1개 호출: qwen3-14b tool 88%, 인자 64%로 1위.

Parallel (12턴) — tool 2개 동시 호출: 전 모델 실패 (최고 인식률 50%).
실서비스에서는 1개씩 순차 분리 호출 필수.

No-Call (12턴) — tool 안 불러야 정답:

| 모델 | 미호출 정답 | Tool Acc | 성향 |
|------|-----------|----------|------|
| llama-3.3-70b-instruct | 0% | 86% | tool 과잉 |
| mistral-small-3.2-24b-ins | 92% | 43% | tool 부족 |
| qwen3-14b | 33% | 83% | tool 과잉 |
| qwen3-next-80b-a3b-instru | 83% | 78% | 균형 |

tool 과잉 모델은 프롬프트 보강으로 개선 가능.

### 7.5 스트레스 민감도 — 시나리오 구분 유의미성

| 모델 | ST1(조건누적) | ST2(맥락희석) | ST3(교란주입) | 최약점 |
|------|-------------|-------------|-------------|--------|
| llama-3.3-70b | 68% | 65% | 67% | ST2 |
| mistral-small-3.2-24b | 53% | 63% | 38% | ST3 |
| qwen3-32b | 69% | 74% | 74% | ST1 |
| qwen3-14b | 71% | 75% | 80% | ST1 |
| qwen3-next-80b | 78% | 81% | 70% | ST3 |

3축 분리의 유의미성:

- 스트레스 유형 간 최대 편차 25.5%p (mistral ST2 63% vs ST3 38%) → 3축 분리가 실질적으로 다른 실패 모드를 포착
- 반면 콜 유형(O1 청약 vs O2 보류)은 평균 3.1%p 차이로 미미 → 모델 성능에 유의미한 영향 없음
- 결론: 시나리오 구분에서 진짜 변별력을 가지는 것은 스트레스 유형(ST1/ST2/ST3)

### 7.6 인사이트 종합

1) tool 과잉 성향과 교란 내성의 trade-off

qwen3-14b는 교란(ST3)에 가장 강하고(80%), 조건누적(ST1)에 가장 약하다(71%).
이는 tool 과잉 성향(No-Call 33%) 때문이다 — 교란 후에도 주저 없이 tool을 호출하므로 ST3에서 유리하지만,
불필요한 호출을 억제하지 못해 No-Call에서 손해를 본다.

프롬프트로 No-Call 판별만 보강하면, 교란 내성은 유지하면서 No-Call 정확도를 높일 수 있다.

2) 민감도 분석 — 어디를 고치면 Performance가 가장 오르는가

| 지표 | 현재 | 여유 | +10%p당 Perf 효과 | 최대 효과 | 우선순위 |
|------|------|------|-------------------|---------|---------|
| Arg Acc | 80% | +15%p | +2.9%p | +4.2%p | 1 |
| No-Call | 33% | +52%p | +1.4%p | +7.4%p | 1 |
| Tool Acc | 93% | +5%p | +2.9%p | +1.4%p | 2 |

**Arg Acc와 No-Call이 가장 효과적인 개선 레버**.

3) H100 1대 환경에서의 최적 선택

| 모델 | 실무 Perf | FP16 VRAM | H100 80GB 탑재 | 비고 |
|------|----------|-----------|---------------|------|
| qwen3-14b | 82% | ~28GB | FP16 가능 (52GB 여유) | Agent 1위, NL 1위 |
| qwen3-next-80b | 77% | ~160GB | 양자화 필수 | 전체 Perf 1위이나 실무 −5%p |
| qwen3-32b | 76% | ~64GB | INT8 필요 (여유 적음) | 14b 대비 열세 |
| llama-3.3-70b | 68% | ~140GB | 양자화 필수 | Arg Acc 심각 (37%) |

**qwen3-14b가 성능·효율·안정성 모두에서 최적**. FP16 서빙으로 양자화 리스크를 피하면서,
H100 여유 VRAM으로 동시 요청을 처리할 수 있다.

---

## 8. 단기 성능 개선 방향

### 8.1 제약 조건

H100 1대 (폐쇄망), 인력/리소스 제한.
파인튜닝·앙상블은 현실적으로 제외, 프롬프트 + 시스템 아키텍처로 승부한다.

| 방법 | 기대 효과 | 기간 | 난이도 | 리스크 |
|------|----------|------|--------|--------|
| 프롬프트 엔지니어링 | 82% → 90% | 2~4주 | 낮음 | 낮음 (가역적) |
| 시스템 아키텍처 개선 | +5~10%p | 2~4주 | 중간 | 낮음 |
| Tool Description 최적화 | 간접적 전체 향상 | 1~2주 | 낮음 | 낮음 |

### 8.2 프롬프트 엔지니어링 (핵심)

현재 병목 2가지를 프롬프트로 해결하는 단계적 로드맵:

Phase 1 — 85% 목표 (Quick Win, 1~2주)
- No-Call 33% → 56%: 시스템 프롬프트에 no-call 가이드 추가
  - `"필수 정보가 부족하면 tool 대신 고객에게 질문하세요"`
  - slot_question few-shot 2~3개 추가

Phase 2 — 90% 목표 (2~4주)
- Arg Acc 80% → 90%+: Structured Slot Tracking 프롬프트 도입
  - `"고객 정보를 JSON으로 누적 추적하고, tool 호출 시 반드시 해당 JSON에서 인자를 채워 넣으세요"`
- No-Call 56% → 80%+: few-shot 보강 (경계 케이스 추가)

Phase 3 — 92~95% 목표 (고도화)
- Arg Acc → 95%+: Chain-of-Thought 인자 검증 단계 도입
- No-Call → 90%+: 경계 케이스 few-shot 확대

프롬프트 변경 시 주의사항:
프롬프트 수정 시 다른 능력이 저하될 수 있다 ([Tool-Induced Myopia][tim] 현상).
단, No-Call 개선은 불필요한 tool 호출을 줄이는 방향이므로 오히려 긍정적이다.
프롬프트 변경 전후 반드시 전체 벤치마크 재실행 → Tool Acc, Arg Acc, No-Call, NL Qual 전부 확인.
한쪽이 올라가면서 다른 쪽이 떨어지는 회귀를 사전에 잡는다.

### 8.3 시스템 아키텍처 개선 (프롬프트 밖에서 해결)

모델 자체를 바꾸지 않고, 모델 주변 시스템으로 정확도를 보강하는 방법:

| 기법 | 설명 | 개선 대상 |
|------|------|----------|
| Parallel → Sequential 분리 | 병렬 호출을 순차 1개씩 호출로 전환 ([LLMCompiler][llmcompiler]) | Parallel 50% → ~100% |
| Slot Memory (외부 JSON) | 대화에서 추출한 슬롯을 코드로 관리, tool 호출 시 주입 | Arg Acc 하락 방지 |
| [MemTool][memtool] 패턴 | 멀티턴에서 tool context를 동적 관리 (불필요 context 제거) | 장기 대화 성능 유지 |
| Tool Description 동시 최적화 | 프롬프트와 tool 설명을 함께 정제 ([ACL 2025][joint-opt]) | 전체 정확도 |

특히 Slot Memory는 모델에게 인자 추출을 전부 맡기지 않고, 대화에서 추출한 정보를 별도 JSON으로 유지하여
tool 호출 시 코드 레벨에서 인자를 채워주는 방식. Arg Acc를 시스템 수준에서 보장한다.

### 8.4 제외한 방법과 이유

| 방법 | 제외 이유 |
|------|----------|
| LoRA / RL 파인튜닝 | H100 1대로 서빙+학습 동시 불가, 인력 부족, 범용 능력 저하 리스크 |
| 모델 앙상블 / 라우팅 | qwen3-14b가 Agent+답변 겸용 1위, 2모델 운영은 GPU 메모리 초과 |
| RAG (인수규정 주입) | 현재 병목이 정보 부족이 아닌 tool calling 정확도이므로 우선순위 낮음 |

파인튜닝이 필요해지는 시점: 프롬프트만으로 95% 벽을 넘지 못할 때.
그때는 [Anchored SFT][asft] + DPO 방식으로 범용 능력 저하를 최소화해야 한다.

---

## 부록

### A. 실행 로드맵

| Phase | 내용 | 산출물 |
|-------|------|--------|
| 1 | 시나리오 + Tool 설계 | scenarios_6_multi_turn.jsonl, tools_spec.py |
| 2 | 평가 모듈 구현 | evaluator.py |
| 3 | 벤치마크 실행 | run_benchmark.py, results/ |
| 4 | 결과 비교 + 리포트 | compare_results.py, report.txt |
| 5 | 성능 개선 실험 | experiments/ |

### B. 참고 문헌

#### 벤치마크 프레임워크

- [BFCL v4][bfcl] — Berkeley Function Calling Leaderboard ([논문][bfcl-paper], ICML 2025)
- [FunctionChat][functionchat] — 한국어 tool-use 대화 벤치마크 ([논문][functionchat-paper], 2024)
- [τ²-bench][tau-bench] — Dual-Control 환경 대화형 에이전트 벤치마크 ([논문][tau-bench-paper], Sierra, 2025)
- [MCP-Bench][mcp-bench] — Model Context Protocol 기반 실세계 태스크 벤치마크 (NeurIPS 2025)

#### 평가 방법론

- [LLM-as-Judge Survey][llm-judge] — LLM 기반 평가 방법론 종합 서베이 (2024)

#### 시나리오 설계 근거

- [The Psychology and Neuroscience of Forgetting][forgetting] — 망각의 3대 원인 종합 리뷰 (Wixted, Annual Review of Psychology 2004)
- [Lost in the Middle][lost-middle] — LLM이 긴 컨텍스트 중간 정보를 잃는 현상 실증 (Liu et al., 2024)

#### 성능 개선 근거

- [Tool-Induced Myopia (TIM)][tim] — 도구 사용이 추론 능력을 저하시키는 현상 (2025)
- [LLMCompiler][llmcompiler] — Parallel Function Calling 컴파일러 (ICML 2024)
- [MemTool][memtool] — 멀티턴 대화에서 tool context 동적 관리 (2025)
- [Joint Optimization][joint-opt] — 프롬프트 + Tool Description 동시 최적화 (ACL 2025 Findings)
- [Anchored SFT][asft] — KL 정규화로 파인튜닝 시 범용 능력 보존 (2024)

<!-- Reference Links -->
[forgetting]: https://www.annualreviews.org/content/journals/10.1146/annurev.psych.55.090902.141555
[lost-middle]: https://arxiv.org/abs/2307.03172
[bfcl]: https://gorilla.cs.berkeley.edu/leaderboard.html
[bfcl-paper]: https://openreview.net/forum?id=2GmDdhBdDk
[functionchat]: https://github.com/kakao/FunctionChat-Bench
[functionchat-paper]: https://arxiv.org/abs/2411.14054
[tau-bench]: https://github.com/sierra-research/tau2-bench
[tau-bench-paper]: https://arxiv.org/abs/2506.07982
[mcp-bench]: https://github.com/Accenture/mcp-bench
[llm-judge]: https://arxiv.org/abs/2411.16594
[tim]: https://arxiv.org/abs/2511.10899
[llmcompiler]: https://arxiv.org/abs/2312.04511
[memtool]: https://arxiv.org/abs/2507.21428
[joint-opt]: https://aclanthology.org/2025.findings-acl.1149/
[asft]: https://arxiv.org/abs/2509.23753
