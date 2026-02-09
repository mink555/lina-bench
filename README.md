# AI TMR Assistant — Multi-Turn Tool-Calling Benchmark

폐쇄망 MCP 구축을 위한 오픈소스 LLM 선정 벤치마크.
5개 모델의 Agent 역량(tool calling 정확도)과 한국어 답변 생성을 106턴 멀티턴 스트레스 테스트로 비교함.

### Quick Start

```bash
git clone https://github.com/mink555/lina-bench.git && cd lina-bench
pip install -r requirements.txt
cp .env.example .env               # OPENROUTER_API_KEY 입력
python -m benchmark.run_benchmark   # 전체 실행 (seed=42, temp=0.0)
python -m benchmark.compare_results # 리포트 + 차트 생성
```

<br>

## Key Findings

> [!IMPORTANT]
> **qwen3-14b** — 실무 Performance **79%** (5개 모델 중 1위)
>
> - Agent 역량 1위: Tool 88%, Arg 74%, FC 94% (@T7 실무 구간 — [📌 세부 진단](#-1위-모델-실무-구간t7-세부-진단))
> - 답변 품질 1위: NL Quality 100%로 Agent+답변 겸용 가능
> - H100 80GB 최적: 14B ~28GB FP16 서빙, 52GB 여유
> - 성능 붕괴: ~T3까지 85%+ 유지, 이후 점진 하락
> - 핵심 병목: No-Call 44%, Arg Acc 74% — [개선 방향](#5-성능-개선-방향) 참고

<br>

---

## 1. 배경 및 목적

폐쇄망 환경에서 MCP(Model Context Protocol)를 구축하기 위해 자체 호스팅 가능한 오픈소스 LLM을 선정해야 함.
모델 "성능"을 2가지 축으로 정의함.

- **Agent 역량 (핵심)** — tool calling 정확도: 호출 정확도(단일/병렬), 인자 정확도, 행동 판단(호출/미호출)
- **한국어 답변 생성 (서브)** — 자연어 응답 품질: 관련성·정확성·간결성·적절성

Agent 역량이 핵심인 이유: MCP의 핵심 가치는 LLM이 외부 도구를 정확히 호출하는 것임.
tool을 잘못 호출하거나 인자를 틀리면 후속 파이프라인 전체가 실패함.
NL Quality는 서브지표이나, 실제 서비스로 이어질 때 고객 응대 품질로 직결되므로 함께 관리함.

MCP 구축과 동시에 AI TMR(텔레마케터) Assistant 챗봇 PoC를 수행해야 함.
TMR이 고객과 영업콜을 하면서 실시간으로 챗봇에 질문하는 상황을 가정하며,
핵심 질문은 **"몇 turn까지 모델이 정확하게 tool을 호출할 수 있는가?"** 임.

모델이 몇 턴에서 성능이 붕괴(85% 이하)되는지 파악하면,
PoC 개발 시 허용 가능한 턴 수와 아키텍처 설계의 근거를 확보할 수 있음.

<br>

---

## 2. 벤치마크 프레임워크 선정

모델을 비교하려면 측정 기준이 필요함.
tool calling 정확도를 체계적으로 평가할 수 있는 기존 벤치마크 프레임워크를 리서치함.

| | [BFCL v4][bfcl] | [FunctionChat][functionchat] | [τ²-bench][tau-bench] |
|---|---|---|---|
| 개발 | UC Berkeley | Kakao | Sierra Research |
| 언어 | 영어 | 한국어 | 영어 |
| 측정 초점 | Tool 호출 정밀도 | 상황별 행동 판단 | E2E Task 완수율 |
| 평가 방식 | AST 매칭 | [LLM-as-Judge][llm-judge] | Pass rate |
| Multi-turn | O (v4부터) | O | O |
| Parallel Call | O | — | — |
| No-Call 판단 | △ (일부) | O (Slot Q + Relevance) | — (환경 내장) |
| 환경 구축 | 불필요 | 불필요 | 필요 (tool 백엔드 + user sim) |

**선정: BFCL v4 + FunctionChat** — 하나만 쓰면 빠지는 평가 차원이 있음.

- BFCL v4만 사용 시: No-Call 세분화 없음, 한국어 답변 품질 평가 없음
- FunctionChat만 사용 시: args 수준의 AST 정밀도 없음, parallel call 평가 빠짐
- BFCL v4(호출 정밀도 → Tool Acc, Arg Acc) + FunctionChat(행동 판단 → FC Judge, NL Quality)을 결합함

<details>
<summary>τ²-bench 제외 사유</summary>

[τ²-bench][tau-bench]는 에이전트와 사용자 모두가 공유 환경을 수정하는 Dual-Control E2E 벤치마크임.

- 환경 구축 부담: tool 실행 백엔드(DB + API)와 user simulator를 도메인별로 구축해야 함
- 도메인 불일치: 지원 도메인(airline, retail, telecom)에 보험 TMR이 없어 도메인 재설계 필요
- 현 단계 목표와 불일치: PoC 단계에서는 모델의 raw function calling 정확도를 먼저 검증하는 것이 우선

</details>

<br>

---

## 3. 시나리오 설계

기존 벤치마크에는 보험 TMR 도메인이 없으므로, 실제 TMR 업무를 반영한 시나리오를 직접 설계함.
단순히 "몇 점"만 비교하는 게 아니라 "몇 턴에서 무너지는가", "왜 무너지는가"까지 알아야 실무에 적용할 수 있으므로, 스트레스 테스트도 함께 설계함.

Turn = TMR이 AI 챗봇에 1회 입력 (고객 대화 턴 ≠ 챗봇 턴)이며, TMR은 판단이 필요한 순간에만 챗봇을 사용함.

### 3.1 콜 유형

TMR 영업콜을 결과 기준으로 3가지 유형으로 분류함.

| 콜 유형 | 설명 | 실무 턴 수 | 스트레스 턴 |
|---------|------|-----------|------------|
| O1 청약 | 가입 의사 → 조건 확인·보험료 산출·청약 절차 | ~7턴 | 18~19턴 |
| O2 보류 | 고민/비교 → 추천·비교 후 재연락 약속 | ~5턴 | 16~17턴 |
| ~~O3 실패~~ | 거절 → 즉시 종료 | 2~4턴 | 제외 |

실패콜(O3)은 2~4턴으로 빠르게 종료되어 스트레스 테스트에 필요한 턴 수 확보가 불가능하므로 제외함.

<details>
<summary>실무 턴 수 도출 근거</summary>

실무 턴 수는 공개 데이터셋이 아닌, 보험 TM 표준 콜 스크립트 구조에서 도출한 추정치임.
TMR이 챗봇에 판단을 요청하는 Decision Point 기준으로, 콜 스크립트의 단계 수와 대응함:

| 유형 | 콜 스크립트 단계 (Decision Point) | 턴 수 |
|------|-----------------------------------|-------|
| 청약 | 본인확인 → 니즈파악 → 건강고지 → 상품추천 → 보험료산출 → 특약설명 → 청약안내 | ~7 |
| 보류 | 본인확인 → 니즈파악 → 상품추천 → 비교설명 → 재연락약속 | ~5 |
| 실패 | 본인확인 → 니즈파악 → 재연락약속 → 종료 | 2~4 |

</details>

### 3.2 3축 스트레스 테스트

실무 대비 2.5~3배 턴을 늘려 스트레스를 주되, 스트레스를 한 덩어리로 주면 "왜 틀렸는지"를 알 수 없음.
원인별로 분리해야 개선 방향이 보이므로, 스트레스 원인을 3축으로 독립 분리함.

설계 근거 — 인지심리학의 기억 실패 3대 원인 (MECE, [Wixted 2004][forgetting]).
LLM도 동일한 패턴을 보인다는 점이 [Lost in the Middle][lost-middle] 등에서 실증됨.

| 코드 | 인지적 원인 | 스트레스 | 예시 |
|------|------------|---------|------|
| ST1 | Information Overload | 조건누적 — 매 턴 새 조건 추가 | 고혈압→흡연→갑상선→당뇨 |
| ST2 | Decay over Distance | 맥락희석 — 다양한 토픽 분산 | 입원→특약→Q&A→비교→업셀 |
| ST3 | Interference | 교란주입 — 고객이 번복·전환 | 흡연→비흡연, 6만→3만→5만 |

분리 원칙: ST1은 조건만 쌓임, ST2는 토픽만 분산, ST3은 번복만 (조건 누적 없음).

### 3.3 시나리오 매트릭스

| | ST1 조건누적 | ST2 맥락희석 | ST3 교란주입 |
|---|---|---|---|
| O1 청약 | O1_ST1 (19턴) | O1_ST2 (18턴) | O1_ST3 (19턴) |
| O2 보류 | O2_ST1 (17턴) | O2_ST2 (16턴) | O2_ST3 (17턴) |

총 **106턴** (single 82 + parallel 12 + no_call 12), 20개 tool 전부 사용.

<details>
<summary>시나리오 상세 / 당사 특화 / Parallel·No-Call 설계</summary>

**시나리오 상세**

| 시나리오 | 고객 | 주요 스트레스 | 핵심 테스트 |
|----------|------|-------------|------------|
| O1_ST1 | 45세 남, 건강보험 → 암보험 | 고혈압→흡연→갑상선→당뇨 | T19에서 4개 조건 기억 |
| O1_ST2 | 38세 여, 실손보험 → 입원보장 | 특약→Q&A→비교→예산→업셀 | 토픽 전환 후 초반 맥락 유지 |
| O1_ST3 | 42세 남, 종신보험 → 암보험 | 흡연↔비흡연, 암↔건강, 6만→3만→5만 | state 번복 후 올바른 값 갱신 |
| O2_ST1 | 50세 남, 암보험 → 건강보험 | 당뇨→간→혈압 | 3개 조건 누적 기억 |
| O2_ST2 | 35세 여, 실손보험 → 암보험 | 비교→치아→추천→업셀 | 폭넓은 토픽에서 맥락 유지 |
| O2_ST3 | 47세 남, 건강보험 → 종신보험 | 종신↔간병, 10만→4만→7만 | 상품·예산 이중 번복 처리 |

**당사 특화**

- 치아보험 (충치·임플란트·보철): O2_ST2에서 임플란트 이력 고객
- 간병보험 (뇌질환·장기요양): O2_ST3에서 간병 전환
- 55세 간편보험: O2_ST1에서 유병자 간편심사

**Parallel / Multiple Function Call**

[BFCL v4][bfcl]의 Parallel(다른 tool 동시) + Parallel-Multiple(같은 tool, 다른 args)을 반영.
시나리오당 2개 × 6 = 12개 parallel turn. Single과 별도 스코어 산출.

**No-Call 턴: Slot Question + Relevance Detection**

[FunctionChat][functionchat]의 Slot Question(추가 질문) + Relevance Detection(범위 밖 거부)을 반영.
시나리오당 2개 × 6 = 12개 no_call turn.

- Slot Question: 필수 정보 누락 시 tool 대신 질문
- Relevance Detection: 보험 외 요청(적금·대출 등)을 tool 없이 거절

**Turn-Point 평가**

1개의 긴 시나리오를 T3·T5·T7·T10·T13·T15·T17·T19 지점에서 슬라이싱하여 성능 곡선을 생성함.

</details>

<br>

---

## 4. 실험 결과 및 분석

위 시나리오로 5개 모델을 실제 실행한 결과임.
"어떤 모델이 가장 좋은가"와 "몇 턴에서 무너지는가"에 답함.

> 106턴 × 5모델 | Judge: GPT-4o | seed=42, temp=0.0 | 실무 기준: ~T7 누적
> | 📄 [상세 리포트](benchmark/results/report_20260209_105911.txt) | 📊 [Markdown 리포트](benchmark/results/report_20260209_105911.md)

### 4.1 종합 성적표

| 모델 | Tool | Arg | FC | NL | 실무 Perf | 전체 Perf |
|------|------|-----|----|----|-----------|-----------|
| llama-3.3-70b-instruct | 85.6% | 36.5% | 84.1% | 0% | 68% | 67% |
| mistral-small-3.2-24b-ins | 8.5% | 7.1% | 45.3% | 32% | 26% | 27% |
| qwen3-32b | 75.0% | 58.0% | 84.4% | 83% | 75% | 72% |
| **qwen3-14b** | **84.0%** | **62.0%** | **88.7%** | **100%** | **79%** | **77%** |
| qwen3-next-80b-a3b-instru | 77.1% | 58.8% | 85.7% | 26% | 77% | 75% |

### 4.2 모델 선정: qwen3-14b

**1위: qwen3-14b** — 실무 Performance **79%**, Agent + 답변 생성 겸용 가능.

#### 📌 1위 모델 실무 구간(@T7) 세부 진단

42턴 기준 (tool_call 36 + no_call 6):

| 지표 | @T7 | |
|------|-----|-|
| Tool Acc | 88% | |
| Arg Acc | 74% | ⚠️ 병목 |
| FC Judge | 94% | |
| No-Call | 44% | ⚠️ 병목 (FC 기반, 전체 nc_acc 33%) |
| **Performance** | **79%** | |

개선 우선순위: **No-Call 44%** > **Arg Acc 74%** > Tool Acc 88%

### 4.3 Turn-Point 분석 — 붕괴 지점

시나리오를 T3~T19 지점에서 슬라이싱하여 누적 Performance를 계산함.
🟢 90%+ | 🔵 85%+ | 🟡 75%+ | 🔴 <75%

| 모델 | ~T3 | ~T5 | ~T7 | ~T10 | ~T13 | ~T15 | ~T19 |
|------|-----|-----|-----|------|------|------|------|
| llama-3.3-70b | 🟡 79% | 🔴 67% | 🔴 68% | 🔴 70% | 🔴 65% | 🔴 66% | 🔴 67% |
| mistral-small-3.2-24b | 🔴 11% | 🔴 20% | 🔴 26% | 🔴 22% | 🔴 28% | 🔴 27% | 🔴 27% |
| qwen3-32b | 🔵 85% | 🔴 71% | 🟡 75% | 🟡 75% | 🔴 72% | 🔴 72% | 🔴 72% |
| **qwen3-14b** | **🔵 89%** | **🟡 80%** | **🟡 79%** | **🟡 81%** | **🟡 77%** | **🟡 76%** | **🟡 77%** |
| qwen3-next-80b | 🔴 74% | 🟡 75% | 🟡 77% | 🟡 76% | 🟡 76% | 🟡 75% | 🟡 75% |

- qwen3-14b는 ~T3까지 🔵 85%+ 유지, 이후 점진적 하락. 실무 7턴 구간에서 79%로 가장 높음
- 붕괴 순서: 인자(Arg) → 도구(Tool) → 행동(FC) — 3/4 모델 공통
- 실무 권장: 7턴 이내 사용 시 79%, 충분히 활용 가능

### 4.4 능력 해부 — Single / Parallel / No-Call

- **Single** (82턴) — qwen3-14b tool 87%, 인자 66%로 1위
- **Parallel** (12턴) — 전 모델 실패 (최고 인식률 58%). 한 문장에 2개 요청이 섞이면 1개만 호출하거나 인자를 섞음 → [서비스 구축 시 해결(D)](#d-parallel-실패--서비스에서-순차-분리)
- **No-Call** (12턴) — 아래 표 참고

| 모델 | 미호출 정답 | Tool Acc | 성향 |
|------|-----------|----------|------|
| llama-3.3-70b-instruct | 0% | 86% | tool 과잉 |
| mistral-small-3.2-24b-ins | 100% | 9% | tool 부족 |
| qwen3-32b | 33% | 75% | tool 과잉 |
| qwen3-14b | 33% | 84% | tool 과잉 |
| qwen3-next-80b-a3b-instru | 83% | 77% | 균형 |

tool 과잉 모델(3개)은 No-Call 정확도가 낮아 불필요한 tool 호출이 발생함.

### 4.5 스트레스 민감도

| 모델 | ST1(조건누적) | ST2(맥락희석) | ST3(교란주입) | 최약점 |
|------|-------------|-------------|-------------|--------|
| llama-3.3-70b | 65% | 67% | 68% | ST1 |
| qwen3-32b | 66% | 78% | 72% | ST1 |
| qwen3-14b | 73% | 76% | 82% | ST1 |
| qwen3-next-80b | 83% | 76% | 68% | ST3 |

- 스트레스 유형 간 최대 편차 15.1%p (qwen3-next-80b ST1 83% vs ST3 68%) → 3축 분리가 실질적으로 다른 실패 모드를 포착함
- 콜 유형(O1 vs O2)은 평균 4.9%p 차이로 미미 → 변별력은 스트레스 유형(ST1/ST2/ST3)에 있음

### 4.6 인사이트 종합

**1) tool 과잉 성향과 교란 내성의 trade-off**

qwen3-14b는 교란(ST3)에 가장 강하고(82%), 조건누적(ST1)에 가장 약함.
tool 과잉 성향(No-Call 33%) 때문 — 교란 후에도 주저 없이 tool을 호출하므로 ST3에서 유리하지만,
불필요한 호출을 억제하지 못해 No-Call에서 손해를 봄.
tool 과잉 성향은 교란 내성에서 유리하나, No-Call 정확도를 희생하는 trade-off가 존재함.

**2) 민감도 분석 — 어디를 고치면 Performance가 가장 오르는가**

| 지표 | 현재 | 여유 | +10%p당 Perf 효과 | 최대 효과 | 우선순위 |
|------|------|------|-------------------|---------|---------|
| Arg Acc | 74% | +21%p | +2.9%p | +5.9%p | 1 |
| No-Call | 44% | +41%p | +1.4%p | +5.8%p | 1 |
| Tool Acc | 88% | +10%p | +2.9%p | +3.0%p | 2 |

Perf 최대 효과 기준으로 **Arg Acc와 No-Call이 가장 큰 레버**임. 단, 확실성과 난이도를 고려한 실행 순서는 [5절](#5-성능-개선-방향) 참고.

**3) H100 1대 환경에서의 최적 선택**

| 모델 | 실무 Perf | FP16 VRAM | H100 80GB | 비고 |
|------|----------|-----------|-----------|------|
| qwen3-14b | 79% | ~28GB | FP16 (52GB 여유) | Agent 1위, NL 1위 |
| qwen3-next-80b | 77% | ~160GB | 양자화 필수 | 실무 −2%p |
| qwen3-32b | 75% | ~64GB | INT8 필요 | 14b 대비 열세 |
| llama-3.3-70b | 68% | ~140GB | 양자화 필수 | Arg 심각 (36%) |

**qwen3-14b가 성능·효율·안정성 모두에서 최적임**.

<br>

---

## 5. 성능 개선 방향

79%에서 더 올리려면? 에러 분석에서 드러난 병목별로 조치를 정리함.
**지금 할 수 있는 것**(벤치마크 재실행으로 검증)과 **서비스 구축 시 반영할 것**(아키텍처 설계)을 구분함.

> [!NOTE]
> 제약: H100 1대 (폐쇄망), 서빙+학습 동시 불가 → **파인튜닝 제외**, 모델 앙상블 제외 (GPU 초과)

### 🔬 지금 실험 가능 — 프롬프트/tool spec 수정 → 벤치마크 재실행

| | 병목 (에러) | 조치 | 기대 효과 | 확실한가? |
|:---:|------------|------|----------|:--------:|
| A | No-Call 33% (FALSE_CALL 8건) | [프롬프트에 미호출 가이드 추가](#a-no-call-과잉-호출--프롬프트-가이드) | Perf **+5.8%p** | 불확실 |
| B | Tool 혼동 (WRONG_TOOL 18건) | [tool 설명문 정제](#b-tool-혼동--description-정제) | WRONG_TOOL 감소 | 불확실 |
| C | Arg 오류 42건 (번복 18건) | [Slot Memory (프롬프트)](#c-arg-오류--slot-memory-프롬프트-방식) | Perf **+5.9%p** | 불확실 |

> [!WARNING]
> A~C는 모두 프롬프트/tool spec 변경이라 **효과가 보장되지 않으며**, 다른 능력을 저하시킬 수 있음 ([Tool-Induced Myopia][tim]).
> 변경 전후 반드시 전체 벤치마크 재실행 → 회귀 검증 필요.

### 🏗️ 서비스 구축 시 반영 — 아키텍처/코드 설계

| | 병목 (에러) | 조치 | 기대 효과 | 확실한가? |
|:---:|------------|------|----------|:--------:|
| D | Parallel 전 모델 실패 | [오케스트레이션에서 순차 분리](#d-parallel-실패--서비스에서-순차-분리) | Tool 67→87%, Arg 37→66% | **확실** |
| E | Arg 번복 미갱신 18건 | [Slot Memory (시스템 코드)](#e-arg-번복--slot-memory-시스템-방식) | ARG_STALE 해소 | 비교적 확실 |
| F | 장기 턴 컨텍스트 포화 | [Tool Context 동적 관리](#f-나중에-tool-context-동적-관리) | 나중에 | - |

---

#### A. No-Call 과잉 호출 → 프롬프트 가이드

**문제**: 정보가 부족하거나 보험 외 요청인데 tool을 호출해버림 (FALSE_CALL 8건, 미호출 정답 33%).
"적금 금리 알려줘" 같은 범위 밖 요청에도 `product_lookup`을 부르는 식.

**조치**: 시스템 프롬프트에 "필수 정보가 부족하면 tool 대신 질문하세요 / 보험 외 요청은 거절하세요" + few-shot 예시 추가 → 벤치마크 재실행으로 검증.

**기대 효과**: No-Call 여유분 +41%p × 민감도 1.4%p = **Perf 최대 +5.8%p**.
단, 프롬프트만으로 모델이 판단을 바꿀지 보장 없음 — 실험 필수.

<details>
<summary>근거 / 리스크</summary>

- **근거**: [BFCL v2][bfcl]가 irrelevance detection을 별도 평가 카테고리로 분리한 이유 자체가, 모델이 이 판단을 못하기 때문. [ToolACE][toolace] (ICLR 2025)는 파인튜닝으로 irrelevance 83→90% 개선했으나 프롬프트만으로는 미검증
- **리스크**: [Tool-Induced Myopia][tim] (2025) — tool 관련 프롬프트 강화 시 추론 능력 저하 실증. 벤치마크 재실행 필수

</details>

---

#### B. Tool 혼동 → Description 정제

**문제**: 정답이 아닌 유사한 tool을 호출 (WRONG_TOOL 18건).
예: `rider_detail`을 불러야 하는데 `rider_recommendation`을 부르는 식 — 이름과 설명이 비슷해서 혼동.

**조치**: tool 설명문(description)의 역할 구분을 명확히 하고, 프롬프트에 tool 선택 기준을 보강 → 벤치마크 재실행으로 검증.
A(No-Call 가이드)와 동시 실험 가능 — FALSE_CALL 8건도 함께 대상.

**기대 효과**: WRONG_TOOL + FALSE_CALL = 26건 감소 대상.
역시 효과 보장 없음 — description 변경이 기존 정상 호출을 깨뜨릴 수도 있음.

<details>
<summary>근거 / 리스크</summary>

- **근거**: [PLAY2PROMPT][joint-opt] (ACL 2025) — tool과 직접 상호작용하여 tool instruction 자동 최적화, 라벨 데이터 없이 zero-shot 가능
- **리스크**: A와 동일. tool description 변경이 기존 정상 호출에 영향 가능

</details>

---

#### C. Arg 오류 → Slot Memory (프롬프트 방식)

**문제**: tool은 맞게 골랐지만 인자 값을 틀림 (ARG_WRONG 24건 + ARG_STALE 18건 = 42건).
특히 ARG_STALE 18건은 고객이 "6만원 → 3만원"으로 번복했는데 이전 값(6만원)으로 tool을 호출한 실수.

**조치**: 프롬프트에 "고객 정보를 JSON으로 정리한 뒤 tool 호출하라" 지시 추가 → 벤치마크 재실행으로 검증.

**기대 효과**: Arg Acc 여유분 +21%p × 민감도 2.9%p = **Perf 최대 +5.9%p**.

**한계**: 번복 미갱신(ARG_STALE)은 모델이 못 하는 것을 다시 모델에게 시키는 셈 — 효과 불확실.
JSON 출력 강제가 추론 정확도를 떨어뜨릴 수도 있음 ([Tam et al.][json-degradation]: 최대 −27%p).
→ 프롬프트로 안 되면 서비스 단계에서 시스템 방식(E)으로 전환.

<details>
<summary>근거</summary>

- [State-Update Prompting][state-update] (2025) — 대화 상태를 프롬프트에 명시적으로 재구성하는 전략. 단순 히스토리 연결 대비 **+14.1%** 성능 향상, 추론 시간 73.1% 감소, 토큰 59.4% 절감
- 단, 모델이 상태를 정확히 재구성한다는 전제 — 번복/교란이 심한 경우 모델이 상태를 잘못 구성할 수 있음

</details>

---

#### D. Parallel 실패 → 서비스에서 순차 분리

**문제**: 사용자가 한 문장에 2개 요청을 담으면("A 보험료랑 B 보험료 둘 다 알려줘"), 모델이 2개를 각각 호출해야 하는데 실패함.
1개만 호출하거나, 2개를 호출해도 인자를 섞음. 전 모델 공통 (최고 인식률 58%, Arg 37%).

**원인**: 병렬 호출은 "어떤 요청이 몇 개인지 파싱 → 각각 독립 호출"을 한 번에 해야 해서, 모델에게 근본적으로 어려운 형식임.
프롬프트나 파인튜닝으로 개선한 사례가 거의 없고, 모델 능력의 한계에 가까움.

**조치 (서비스 구축 시)**: MCP 챗봇의 오케스트레이션 레이어에서 병렬 요청을 감지해 1개씩 순차로 모델에 전달.
현재 벤치마크에서는 "모델이 parallel을 못 한다"를 확인한 단계이며, 실제 분리 로직은 서비스 설계 시 반영.
모델 자체는 건드리지 않으므로 회귀 리스크 0.

**기대 효과**: Parallel 턴의 Tool 67→87%, Arg 37→66% (본 벤치마크 Single 실측치 기준).
단, Parallel은 12/106턴(11%)이라 전체 Perf 기여는 제한적 — 지연도 1→2회 호출로 증가.

<details>
<summary>근거</summary>

- [LLMCompiler][llmcompiler] (ICML 2024) — 병렬 호출을 DAG로 분해 → 순차 실행으로 정확도 ~9%p 향상, 지연 3.7× 감소

</details>

---

#### E. Arg 번복 → Slot Memory (시스템 방식)

C(프롬프트 방식)로 ARG_STALE이 해결되지 않을 경우의 대안.

**조치 (서비스 구축 시)**: 별도 코드가 대화에서 슬롯(고객 나이, 예산, 조건 등)을 추출·관리하고, tool 호출 시 인자를 코드로 주입.
모델에게 상태 추적을 맡기지 않으므로 **모델의 장기 기억 한계를 우회**.

**확실성**: 비교적 확실 — 상태 관리가 결정적(deterministic). 모델 프롬프트 변경 없이 가능하므로 회귀 리스크 낮음.

<details>
<summary>근거</summary>

- [HammerBench][hammerbench] (2024) — 외부 정보 개입 시 slot-filling 정확도 유의미 하락 (GPT-4o: 74→66%). 모델에게 slot 추적을 전부 맡기는 것 자체가 병목
- [Zero-shot Slot Filling][zs-slot] (COLING 2025) — 경량 추출 모델(GLiNER) + LLM + 규칙 기반 제약을 조합한 파이프라인이 LLM 단독 대비 **+34%** 정확도 향상

</details>

---

#### F. (나중에) Tool Context 동적 관리

실무 7턴에서는 컨텍스트 포화가 발생하지 않음. 서비스에서 턴 제한을 늘릴 때 검토.

<details>
<summary>근거</summary>

- [MemTool][memtool] (2025) — 불필요한 tool context를 동적 제거/교체하여 컨텍스트 포화 방지. 13개 LLM, 100회 연속 대화에서 Workflow/Hybrid 모드가 tool 제거 효율 90-94% 달성

</details>

<details>
<summary>제외 방법: 파인튜닝, 모델 앙상블, RAG</summary>

| 방법 | 제외 이유 |
|------|----------|
| 파인튜닝 (LoRA/RL) | H100 1대로 서빙+학습 동시 불가. [RLHF/SIFT는 멀티턴 능력을 오히려 저하시킬 수 있음][mint]. 필요 시 [Anchored SFT][asft] + DPO로 범용 능력 저하 최소화 |
| 모델 앙상블/라우팅 | 2모델 동시 운영은 GPU 메모리 초과 |
| RAG (인수규정 주입) | 현재 병목이 정보 부족이 아닌 tool calling 정확도 |

</details>

<br>

---

## 부록

<details>
<summary>A. 평가 지표 상세</summary>

### 지표 요약

| 축 | 지표 | 출처 | 집계 범위 | 측정 내용 |
|---|------|------|----------|----------|
| Agent (핵심) | Tool Acc | BFCL v4 | tool_call 턴만 | 정답 tool 선택 + 인자 정확도 |
| Agent (핵심) | FC Judge | FunctionChat | 전체 턴 | 행동 판단 (호출/미호출/개인정보) |
| 답변 (서브) | NL Quality | LLM-as-Judge | 텍스트 있는 턴 | 자연어 응답 품질 (GPT-4o) |

### Tool Acc — BFCL Score (tool_call 턴 전용)

| Sub-metric | 설명 | 비고 |
|------------|------|------|
| tool_name_acc | 정답 tool 이름 exact match | |
| arg_key_acc | GT key 존재 비율 | tool name 정답일 때만 산출 |
| arg_value_acc | key-value deep compare | tool name 정답일 때만 산출 |

- tool name이 틀리면 arg = 0 ([BFCL v4 원본 방식][bfcl-paper])
- Parallel: greedy 매칭 후 개별 평가, `parallel_detected` = 복수 호출 인식
- no_call 턴은 BFCL 집계에서 완전 제외 — 미호출 판단은 FC Judge가 전담
- Multi-turn Memory: 후반 GT args에 앞 턴 조건이 모두 포함, 하나라도 잊으면 arg 하락

### FC Judge — 행동 판단 (전체 턴)

| Sub-metric | Tool Call 턴 | No-Call 턴 |
|------------|-------------|-----------|
| action_type_acc | 호출=1, 미호출=0 | 미호출=1, 호출=0 |
| tool_selection_acc | 이름 일치=1 | 미호출=1, 호출=0 |
| privacy_detection_acc | privacy 턴 정답=1 | 항상 1.0 |

### NL Quality (서브)

GPT-4o가 관련성·정확성·간결성·적절성을 Pass/Fail 판정함 ([LLM-as-Judge][llm-judge] 방식).
현 단계에서는 참고 지표, 향후 서비스 구축 시 핵심 지표로 승격 가능.

### Performance — 종합 점수

```
tool_call 턴:  (Tool Acc + Arg Acc + FC Judge) / 3
no_call 턴:    FC Judge만
전체 Performance = 모든 턴의 per-turn 점수 평균 (micro-average)
```

### 지표 무결성 설계

```
                     ┌─ tool_call 턴 ─┐   ┌─ no_call 턴 ─┐
  Tool Acc (BFCL)    │    산출         │   │   제외        │
  Arg Acc  (BFCL)    │    산출         │   │   제외        │
  FC Judge           │    산출         │   │   산출        │
  NL Quality         │    산출         │   │   산출        │
  Performance        │  (T+A+FC)/3    │   │   FC만        │
                     └────────────────┘   └──────────────┘
```

<details>
<summary>채점 예시: tool_call 턴 vs no_call 턴</summary>

**tool_call 턴 예시** — O1_ST1 시나리오 T5:

```
TMR 입력: "45세 남성, 고혈압 있고 흡연자인데 암보험 보험료 알려줘"

모델 응답: premium_calculator(age=45, gender="male", product="암보험",
           conditions=["고혈압","흡연"])
         + "예상 보험료는 월 8만원입니다"
```

| 지표 | 채점 | 설명 |
|------|------|------|
| Tool Acc | 1.0 | `premium_calculator` 정답 |
| Arg Acc | 0.8 | 예: 흡연 조건을 빠뜨렸다면 key 누락 |
| FC Judge | 1.0 | tool을 불러야 하는 턴에서 불렀고, 이름도 정답 |
| NL Quality | Pass | GPT-4o 판정: 관련성·정확성·간결성 충족 |
| **Performance** | **(1.0 + 0.8 + 1.0) / 3 = 0.93** | |

---

**no_call 턴 예시** — O1_ST1 시나리오 T8 (Relevance Detection):

```
TMR 입력: "이 고객 적금 금리도 좀 알아봐줘"  ← 보험 외 요청

정답 행동: tool을 호출하지 않고 "보험 상담만 가능합니다" 류의 답변
```

| 지표 | 채점 | 설명 |
|------|------|------|
| Tool Acc | — | 제외 (no_call 턴은 BFCL 대상 아님) |
| Arg Acc | — | 제외 |
| FC Judge | 1.0 | tool을 안 불러야 하는 턴에서 안 불렀으므로 정답 |
| NL Quality | Pass | GPT-4o 판정: 범위 밖 요청을 적절히 거절 |
| **Performance** | **FC만 = 1.0** | |

만약 모델이 `product_lookup(keyword="적금")`을 호출했다면 → FC Judge 0.0, Performance도 **0.0**.

</details>

</details>

<details>
<summary>B. Tool 설계서 (20개)</summary>

### 설계 원칙

- MECE: 4개 카테고리 20개 tool, 기능 중복 없음
- 당사 특화: 치아(충치·임플란트·보철), 간편보험, 간병보험
- TMR 현실 반영: CRM 없음, 본인 보유 상품만 조회, 개인정보 위반 차단

### Function 분류

| 카테고리 | 수 | 대표 tool | 역할 |
|----------|---|-----------|------|
| F1 정보조회 | 10 | product_lookup, coverage_detail_lookup, underwriting_rules_lookup | 상품·보장·인수 정보 |
| F2 판단/계산 | 4 | underwriting_eligibility_checker, premium_calculator | 가입 가능성·보험료 |
| F3 추천 | 4 | product_recommender, budget_optimizer | 니즈/예산 기반 추천 |
| F4 규제 | 2 | compliance_checker, privacy_violation_detector | 규정·개인정보 |

### 개인정보 보호

- 허용: 본인 보유 보험, 거절 이력
- 차단: 콜·문의·납입·청구 이력, 의료 기록, 타인 정보

</details>

<details>
<summary>C. 실험 환경</summary>

### 대상 모델

| 모델 | 개발사 | 파라미터 | 아키텍처 | FP16 VRAM |
|------|-------|---------|---------|-----------|
| llama-3.3-70b-instruct | Meta | 70B | Dense | ~140GB |
| mistral-small-3.2-24b-instruct | Mistral | 24B | Dense | ~48GB |
| qwen3-32b | Alibaba | 32B | Dense | ~64GB |
| qwen3-14b | Alibaba | 14B | Dense | ~28GB |
| qwen3-next-80b-a3b-instruct | Alibaba | 80B (3B active) | MoE | ~160GB |

선정 기준: tool calling 지원, 한국어 성능, 오픈소스, 다양한 파라미터 규모(14B~80B).

### 환경 제약

- GPU: H100 80GB × 1대 (폐쇄망)
- 제한: 서빙과 학습 동시 수행 불가 → 파인튜닝 배제
- 배포 조건: FP16 서빙이 가능한 모델이 유리
- 벤치마크 실행: OpenRouter API를 통해 원본 모델의 순수 성능을 측정

### 프로젝트 구조

```
my_bench/
├── README.md
├── .env                          ← OPENROUTER_API_KEY (.env.example 참고)
├── .env.example                  ← 환경변수 템플릿
├── requirements.txt
├── configs/
│   └── default.json              ← 실험 파라미터 (seed, temperature 등)
├── scenarios/
│   ├── scenarios_6_multi_turn.jsonl  ← 6개 시나리오 (106턴)
│   ├── create_sc.py
│   └── stress_test_summary.png
├── tool_specs/
│   └── tools_spec.py             ← 20개 tool 명세서
├── benchmark/
│   ├── evaluator.py              ← 평가 모듈
│   ├── run_benchmark.py          ← 벤치마크 실행기
│   ├── compare_results.py        ← 결과 비교 + 리포트
│   └── results/
│       ├── detail_{run_id}.json      ← 전체 턴별 상세 (기존)
│       ├── summary_{run_id}.json     ← 모델별 집계 (기존)
│       ├── turn_level_{run_id}.jsonl ← 턴별 플랫 레코드
│       ├── scenario_summary_{run_id}.csv ← 시나리오별 CSV
│       ├── report_{run_id}.txt       ← 텍스트 리포트
│       └── report_{run_id}.md        ← Markdown 리포트 (GitHub 렌더링용)
└── experiments/                  ← 성능 개선 실험
```

### 실행 방법

```bash
cp .env.example .env                        # API 키 설정
python -m benchmark.run_benchmark           # 전체 실행 (configs/default.json 사용)
python -m benchmark.run_benchmark --dry-run # Dry-run (API 호출 없이 구조 검증)
python -m benchmark.run_benchmark --config configs/custom.json  # 커스텀 config
python -m benchmark.compare_results         # 결과 비교 + 리포트
```

</details>

<details>
<summary>D. 실행 로드맵</summary>

| Phase | 내용 | 산출물 |
|-------|------|--------|
| 1 | 시나리오 + Tool 설계 | scenarios_6_multi_turn.jsonl, tools_spec.py |
| 2 | 평가 모듈 구현 | evaluator.py |
| 3 | 벤치마크 실행 | run_benchmark.py, results/ |
| 4 | 결과 비교 + 리포트 | compare_results.py, report.txt |
| 5 | 성능 개선 실험 | experiments/ |

</details>

### 참고 문헌

- [BFCL v4][bfcl] — Berkeley Function Calling Leaderboard ([논문][bfcl-paper], ICML 2025)
- [FunctionChat][functionchat] — 한국어 tool-use 대화 벤치마크 ([논문][functionchat-paper], 2024)
- [τ²-bench][tau-bench] — Dual-Control 환경 대화형 에이전트 벤치마크 ([논문][tau-bench-paper], Sierra, 2025)
- [MCP-Bench][mcp-bench] — MCP 기반 실세계 태스크 벤치마크 (NeurIPS 2025)
- [LLM-as-Judge Survey][llm-judge] — LLM 기반 평가 방법론 서베이 (2024)
- [Wixted 2004][forgetting] — 망각의 3대 원인 종합 리뷰 (Annual Review of Psychology)
- [Lost in the Middle][lost-middle] — LLM 긴 컨텍스트 중간 정보 손실 (Liu et al., 2024)
- [Tool-Induced Myopia][tim] — 도구 사용이 추론 능력을 저하시키는 현상 (2025)
- [LLMCompiler][llmcompiler] — Parallel Function Calling 컴파일러 (ICML 2024)
- [MemTool][memtool] — 멀티턴 tool context 동적 관리 (2025)
- [PLAY2PROMPT][joint-opt] — Zero-shot Tool Instruction 최적화 (ACL 2025)
- [ToolACE][toolace] — LLM Function Calling 학습 데이터 생성 + irrelevance detection (ICLR 2025)
- [HammerBench][hammerbench] — 모바일 환경 Function Calling 평가, External Information 영향 분석 (2024)
- [State-Update Prompting][state-update] — 멀티턴 대화 상태 재구성 전략, +14.1% 성능 향상 (2025)
- [Zero-shot Slot Filling][zs-slot] — GLiNER + LLM + 규칙 파이프라인, 단독 대비 +34% (COLING 2025)
- [Tam et al.][json-degradation] — JSON 출력 강제 시 추론 정확도 최대 −27%p 하락 (2024)
- [MINT][mint] — 멀티턴 상호작용 평가, RLHF/SIFT 멀티턴 저하 실증 (Wang et al., 2023)
- [Anchored SFT][asft] — 파인튜닝 시 범용 능력 보존 (2024)

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
[joint-opt]: https://sls.csail.mit.edu/publications/2025/WFang_ACL_2025.pdf
[toolace]: https://proceedings.iclr.cc/paper_files/paper/2025/file/663865ea167425c6c562cb0b6bcf76c7-Paper-Conference.pdf
[hammerbench]: https://arxiv.org/abs/2412.16516
[state-update]: https://arxiv.org/abs/2509.17766
[zs-slot]: https://aclanthology.org/2025.coling-industry.59/
[json-degradation]: https://arxiv.org/abs/2408.02442
[mint]: https://arxiv.org/abs/2309.10691
[asft]: https://arxiv.org/abs/2509.23753
