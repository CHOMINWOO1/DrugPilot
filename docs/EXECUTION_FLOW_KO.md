# DrugPilot 연구 실행 코드 흐름

이 문서는 웹 화면에서 **연구 실행**을 누른 순간부터 Human Gate까지 실제로 호출되는
코드 경로를 설명한다. 아래의 “에이전트”는 각각 독립된 입력·출력 스키마와 책임을 가진
Python 컴포넌트이며, 전체 순서와 재실행 분기는 LangGraph가 통제한다.

## 한눈에 보는 실행 순서

```text
브라우저 researchForm.submit
  → POST /api/runs
  → ResearchBrief 생성·검증 + GoalFocus 파생·SQLite 저장
  → 백그라운드 LangGraph 실행
  → validate_brief
  → MA1 (표적·근거·가설 포트폴리오)
  → MA3/G0 (가설 감사)
      ├─ PASS    → Planner
      ├─ REVISE  → MA1 재실행
      ├─ REPLAN  → MA1에서 가설 계획 재구성
      ├─ ABSTAIN → 중단·사람 검토
      └─ BLOCK   → 실행 차단
  → Planner (실행 가능한 PlanSpec)
  → MA2 (단일 경로 선택 후 후보 생성/재창출·도구 계산·순위화)
  → MA3/G1 (후보·과정 감사)
      ├─ PASS    → Human Gate
      ├─ REVISE  → MA2 선택 재실행
      ├─ REPLAN  → Planner부터 재실행
      ├─ ABSTAIN → 중단·사람 검토
      └─ BLOCK   → 실행 차단
  → HUMAN_GATE 상태 저장 후 LangGraph 실행 종료
  → 연구자 APPROVE / HOLD / REJECT 별도 제출·저장
```

## 1. 브라우저에서 요청 만들기

- 화면: `src/drugpilot/web/templates/index.html`
  - `#researchForm`: 질환, 연구목표, 세포 유형, 후보 수, 금지약물 등의 입력 폼
  - `#runButton`: `type="submit"`인 연구 실행 버튼
- 브라우저 로직: `src/drugpilot/web/static/app.js`
  - `init()`가 폼의 `submit` 이벤트를 `createRun()`에 연결
  - `buildCreateRequest()`가 입력값을 JSON 요청으로 변환
  - `createRun()`이 `POST /api/runs`를 호출
  - 실행 ID를 받으면 `/api/runs/{run_id}/events`에 SSE로 연결
  - SSE가 끊기면 `/api/runs/{run_id}`를 1.8초 간격으로 조회하는 polling으로 전환

즉, 버튼이 미리 저장된 결과를 단순히 여는 것이 아니라 새 `run_id`를 만들고 서버의
워크플로를 실제로 시작한다. LLM이 Mock이어도 공개 DB와 MA2 과학 도구 설정은 별개다.
현재 8765 검증 설정은 실제 DrugCentral·ChEMBL 후보, RDKit 및 CPU ADMET-AI를 사용하고,
검증된 표적별 profile이 있을 때만 Vina를 실행한다. 누락 결과를 fixture로 채우지 않는다.

## 2. API가 Research Brief를 만드는 단계

- 엔드포인트: `src/drugpilot/api/app.py`의 `create_run()`
- 입력 스키마: `CreateRunRequest`
- 변환 결과: `ResearchBrief`
  - `ProjectConstraints`: 세포 유형, 작용 방향, 탐색 우선순위, 금지약물, 후보 수
  - `BudgetPolicy`: 토큰·호출·비용 상한
  - `DataSourcePolicy`: 실행에서 허용되는 데이터 소스
  - `EvaluationDataPolicy`: 런타임과 분리된 평가용 데이터셋

API는 이 객체를 `DrugPilotOrchestrator.create_run(brief)`에 전달하고 HTTP 202로 초기
snapshot을 즉시 반환한다. 긴 연구 실행은 HTTP 요청 안에서 기다리지 않고 별도
`asyncio.Task`로 진행된다.

Orchestrator는 이때 `orchestration/goal_focus.py`의 `derive_goal_focus()`로 연구목표를
제한된 명시적 단서(single-cell, expression, network, Pharos, repurposing, novel target,
safety, 명시 표적)로 해석한 `GoalFocus`도 별도 artifact로 저장한다. 이는 자유문장을
마음대로 추론하는 단계가 아니며, 알 수 없는 문구는 기존 동작을 바꾸지 않는다.

## 3. Orchestrator와 LangGraph

- 핵심 파일: `src/drugpilot/orchestration/graph.py`
- 핵심 클래스: `DrugPilotOrchestrator`
- 그래프 생성: `_build_graph()`
- 실제 실행: `_execute()`의 `self.graph.ainvoke(...)`

`_build_graph()`는 다음 LangGraph 노드를 등록한다. 여기서 `human_gate`는 프로세스
안에서 계속 대기하는 interrupt가 아니다. 검토 가능한 상태를 SQLite에 저장하고 이벤트
스트림을 닫은 뒤 그래프가 끝나며, 연구자 결정은 이후 별도 API 요청으로 기록된다.

| 노드 | 실행 함수 | 역할 |
|---|---|---|
| `validate_brief` | `_validate_brief()` | 입력 스키마·정책 검사 |
| `ma1` | `_run_ma1()` | 근거 기반 가설 포트폴리오 생성 |
| `intake_guard` | `_guard_disease_intake()` | 외부 질환 ID와 입력 문맥 검증; 불확실하면 G0 전에 ABSTAIN |
| `g0` | `_run_g0()` | 가설의 추적성·반증 가능성 감사 |
| `planner` | `_run_planner()` | 승인 도구·연산·예산을 고정한 PlanSpec 생성 |
| `route_guard` | `_guard_ma2_route()` | 경로 근거가 불충분하면 MA2 전에 ABSTAIN |
| `ma2` | `_run_ma2()` | G0 승인 표적별로 재창출 또는 de novo 한 경로씩 실행 |
| `g1` | `_run_g1()` | 후보·안전성·도구 trace·정책 감사 |
| `human_gate` | `_await_human()` | HUMAN_GATE 상태를 저장하고 이번 그래프 실행 종료 |
| `abstain` | `_abstain()` | 근거 부족·상한 도달 시 보류 |
| `block` | `_block()` | 안전·출처·정책 위반 시 차단 |

Orchestrator는 과학적 내용을 직접 생성하는 에이전트가 아니다. 상태 전이, 허용 도구,
호출 예산, 재시도 횟수, 산출물 저장, 감사 결과에 따른 다음 노드를 결정하는 제어 계층이다.

## 4. MA1: 질환에서 가설 포트폴리오까지

- 에이전트: `src/drugpilot/agents/ma1.py`의 `MA1Agent.run()`
- Orchestrator 진입점: `graph.py`의 `_run_ma1()`

주요 절차는 다음과 같다.

1. 입력 질환을 Open Targets에서 정규화하고 정확한 외부 질환 ID 확인
2. 정확 일치 또는 연구자가 제공한 ID가 아니면 fixture 가설을 만들지 않고 `intake_guard`에서 ABSTAIN
3. 확인된 질환의 연관 표적 후보 및 PubMed ESearch/ESummary/EFetch 문헌 조회
4. Open Targets·UniProt를 이용한 표적 ID 정규화
5. Literature 및 Tbio 트랙의 Proposer/Critic 반복
6. 근거, 반증, 반증시험을 포함한 `HypothesisPacket` 생성
7. GoalFocus가 요청한 single-cell·발현·네트워크 맥락을 소스 라우팅에 반영하고,
   명시 표적은 MA1이 실제 생성한 claim-linked support 가설과 정확히 일치할 때만 우선
8. 연구자 우선순위에 Pharos·TDL·druggability 요청이 있거나 comprehensive 프로필에서
   druggability가 필수이면 Pharos를 트랙별 상위 표적 1건에만 조회
9. Context Router가 상위 표적에 필요한 맥락 DB만 제한 조회
10. OpenAI 키가 있으면 표적·근거·ID·claim confidence를 동결한 채 기전·반증 서술만 보강

기본 `comprehensive` 근거 프로필은 핵심 소스당 최대 8건, 표적당 5개 핵심 소스·60초,
표적 탐색 PubMed 40건, 맥락 소스 트랙당 2표적·36회·90초, MA2 공개 DB 보강은 경로당
상위 후보 최대 5개를 사용한다. 빠른 연결 확인용
`focused`는 종전 소스당 2건·맥락 8회·12초를 유지한다. 예산 초과는 조용히 누락하지 않고
`skipped_call_budget` 또는 `skipped_time_budget`으로 기록한다.

PubMed와 Europe PMC의 전체 초록은 저장하지 않는다. 질환과 표적을 같은 문장에서 직접
연결하는 최대 320자 구문만 source location, 조회 시각, 전체 normalized abstract hash,
excerpt hash와 함께 저장한다. 직접 연결 문장이 없으면 metadata-only context로 남고,
내성·무반응·반대 방향·용량제한 독성 구문은 support가 아니라 counter/context로 분리한다.

미확인 질환, 질환 ID–라벨 불일치, 알려진 질환 문맥과 연구목표·세포유형의 충돌은
`DiseaseIntakeAssessment`에 저장된다. 향후 LLM은 정규화 후보와 설명을 제안할 수 있지만,
외부 확인 상태나 다음 노드 실행 권한을 변경할 수 없다.

공개 DB가 실패하면 오류와 fallback 여부를 provenance에 남긴다. Context Router의 레코드는
`supports=context`로만 저장되어 인과 지지 근거로 자동 승격되지 않는다.

소스 선택은 하나의 LLM이 임의로 수행하지 않는다. MA1 기본 조회, Pharos, 맥락 DB,
MA2 후보 DB에 각각 결정론적 라우팅 규칙이 있으며 Orchestrator가 정해진 단계에서만
호출한다. Planner는 26개 런타임 소스 전체를 검토한 coverage manifest를 저장한다. LLM은
등록된 소스의 우선순위·질의 초점·설명만 제안할 수 있으며 source admission, tool operation,
호출·시간·토큰·비용 예산을 변경할 수 없다. 실제 조회·건너뜀·설정 부족·오류 사유는
실행별 호출 이력에 남는다.

탐색 우선순위 드롭다운이 `auto`일 때 연구목표에 명시된 `drug repurposing` 또는
`novel target` 단서는 Planner의 실제 적용 우선순위를 바꿀 수 있다. 구조화 드롭다운을
직접 지정한 경우에는 그 값이 항상 우선한다. 어느 경우든 MA2 경로를 강제로 지정하지
않으며, 동일한 근거 자격 심사를 통과한 기존 가설 사이의 선택 순서만 바꾼다.

## 5. MA3/G0: 가설 감사와 환류

- 에이전트: `src/drugpilot/agents/ma3.py`의 `audit_g0()`
- Orchestrator 진입점: `graph.py`의 `_run_g0()`

G0는 가설의 evidence ID 연결, 출처, 반증 가능성, 순위 무결성 등을 읽기 전용으로
검사한다. 규칙 엔진이 최종 verdict를 계산하며, 설정된 경우 외부 LLM Reviewer의
우려사항을 추가 입력으로 사용할 수 있다. LLM이 PASS를 직접 결정하지 않는다.

`_route_g0()`가 다음 노드를 결정한다. PASS는 Planner, REVISE/REPLAN은 허용 횟수 내에서
MA1, ABSTAIN은 보류, BLOCK은 차단으로 이동한다.

## 6. Planner: 설명이 아니라 실행 계약 만들기

- 에이전트: `src/drugpilot/agents/planner.py`의 `PlannerAgent.run()`
- Orchestrator 진입점: `graph.py`의 `_run_planner()`

Planner는 감사 통과 가설을 `PlanSpec`으로 바꾼다. PlanSpec에는 task ID, 의존성,
허용 tool·operation, 모델, 토큰, 재시도, 합격 기준, 중단 조건이 들어간다. LLM이 연결돼도
실행 계약은 결정론적 PlanSpec이 권위 원본이며, LLM은 설명 문구만 덧붙일 수 있다.

## 7. MA2: 표적 포트폴리오별 경로와 실제 과학 도구 실행

- 에이전트: `src/drugpilot/agents/ma2.py`의 `MA2Agent.run()` / `rework()`
- Orchestrator 진입점: `graph.py`의 `_run_ma2()`

MA2는 G0를 통과하고 식별자가 해석된 각 표적에 대해 약물 근거 강도를 다시 계산하고,
표적마다 `repurposing` 또는 `de_novo` 중 정확히 한 경로를 선택한다. 경로 근거가 부족한
표적은 실행하지 않고 사유와 함께 남긴다. 모든 실행 가능 표적의 검색·평가가 완료되거나
근거 있는 0건 검색으로 종결돼야 포트폴리오가 완료된다.

재창출 경로에서는 먼저 사용자 금지목록을 확인하고, 설정된 DrugCentral·DrugBank의
후보 식별정보·규제상태·해당 질환 금기 레코드를 조회한다. 회수·금지 상태, 정확히 일치하는
금지 약물명/ID/계열, 질환 일치 금기 레코드는 구조 표준화·docking·ADMET 전에 제외한다.
설정된 소스의 전송 오류는 안전하다는 뜻으로 바꾸지 않고 `unresolved`로 보존하여 G1이
ABSTAIN하도록 한다. 소스가 설정되지 않은 기본 Mock 실행은 `not_configured`를 기록한 채
계속 실행하며, 빈 금기 검색 결과 역시 임상적 안전성 증명으로 표시하지 않는다.

재창출은 DrugCentral exact-target 레코드와 ChEMBL의 exact-target 실측 리간드를 합쳐
조회한다. 단위가 명시된 표적 연계 활성은 그대로 보존하고, 구조는 실제 PubChem/ChEMBL
식별자와 함께 추적한다. 직접 측정 활성이 있으면 도킹은 `SKIPPED`로 명시하며 점수는
`null`이다. 그렇지 않은 후보는 production-ready 표적별 receptor/box profile이 있을 때만
Vina를 실행한다. profile이 없으면 후보를 제외하되 CPU ADMET-AI 6종은 실제 실행하여 기록한다.
de novo는 ChEMBL 실측 리간드를 seed로 한 RDKit BRICS 생성이며, 실측 seed나 필수 도구가
없으면 중단한다. 어떤 경우에도 LLM은 docking·ADMET 수치를 작성하거나 덮어쓸 수 없다.
구조/식별정보가 바뀌는 재작업은 동일한 제한소스 사전검사를 다시 거친다. 현재 두 제한
소스는 서로 다른 API/로컬 스키마라 후보별 bounded query를 사용하며, 어댑터 캐시로 동일
질의의 후순위 보강 호출만 재사용한다. DrugBank의 필드·지역 범위는 라이선스에 따르고,
금기 레코드의 부재는 안전성의 음성 근거가 아니라 단지 조회된 레코드가 없다는 뜻이다.

## 8. MA3/G1: 후보뿐 아니라 실행 과정 감사

- 에이전트: `src/drugpilot/agents/ma3.py`의 `audit_g1()`
- Orchestrator 진입점: `graph.py`의 `_run_g1()`

G1은 구조·단위·방향·ADMET 누락·금지약물·제한소스 조회상태·경로 적합성뿐 아니라 PlanSpec과 실제
tool name, operation, task ID, 후보 연결, 실행 순서도 대조한다. `_route_g1()`은
PASS→Human Gate 상태 저장, REVISE→MA2, REPLAN→Planner, ABSTAIN/BLOCK→종료로
분기한다.

## 9. Human Gate와 저장

- 화면 제출: `app.js`의 `submitHumanDecision()`
- API: `app.py`의 `/api/runs/{run_id}/human-decision`
- 서버 검증: `graph.py`의 `apply_human_decision()`

그래프가 HUMAN_GATE 상태를 저장한 뒤 `POST /api/runs/{run_id}/human-decision`이 별도로
`apply_human_decision()`을 호출한다. APPROVE는 후보를 한 건 이상 선택해야 하며, 서버가 후보 존재 여부, 제외 상태,
정책 적격성, 금지약물, 제한소스 `excluded`/`unresolved` 상태, 선택 경로를 다시 검사한다. APPROVE는 `COMPLETED`, HOLD는
`ABSTAINED`, REJECT는 `BLOCKED`로 저장된다.

저장 계층은 `src/drugpilot/storage/sqlite.py`의 `SQLiteRunStore`다.

- `RunSnapshot`: 화면에서 읽는 현재 상태
- immutable Artifact: ResearchBrief, HypothesisPacket, GateVerdict, PlanSpec,
  CandidatePacket, HumanGateDecision 원본
- append-only execution event: 단계 전이와 에이전트 handoff
- artifact/event SHA-256 결속: 저장된 감사 원장의 무결성 확인

실시간 화면은 `src/drugpilot/orchestration/events.py`의 메모리 내 `EventBroker`가 같은
실행의 새 이벤트를 SSE로 전달한다. EventBroker 자체는 과거 이벤트를 재전송하지 않는다.
화면 전달이 잠시 끊겨도 SQLite에 저장된 snapshot과 원장이 권위 원본이며, 브라우저는
SSE 장애 시 1.8초 polling으로 현재 snapshot을 다시 조회한다.

## 현재 실제와 Mock의 경계

| 영역 | 현재 동작 |
|---|---|
| LangGraph 상태 전이 | 실제 실행 |
| MA1 공개 DB 조회 | 실제 read-only API, 실패 시 명시적 fallback |
| G0/G1 규칙 감사 | 실제 실행 |
| Planner 계약·예산 통제 | 실제 실행 |
| MA2 재창출 후보 | DrugCentral + ChEMBL 실제 표적 조회 |
| MA2 de novo | ChEMBL 실측 seed + RDKit BRICS; 조건 미충족 시 fail-closed |
| MA2 ADMET | 로컬 pretrained ADMET-AI CPU 추론, 6 endpoint 필수 |
| MA2 docking | production-ready 표적별 Vina profile이 있을 때만 실행; 직접 실측 활성은 생략 기록 |
| LLM | 키가 없으면 fallback 객체, 키가 있으면 허용 범위 내 구조화 보강 |
| 연구자 결정·SQLite 저장 | 실제 저장 |

따라서 현재 빌드는 실제 공개 데이터와 로컬 계산 도구를 사용하는 연구 우선순위 시스템이다.
그럼에도 계산 결과는 효능·안전성 확정이 아니며, 후보 선정 후 실험과 전문가 검토가 필요하다.
