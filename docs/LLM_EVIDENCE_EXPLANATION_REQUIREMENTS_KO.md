# LLM 근거·판정 설명 로그 요구사항

> 상태: 부분 구현 · PubMed/Europe PMC bounded excerpt와 source coverage manifest 완료 ·
> LLM 기반 종합 설명은 API 연동 단계에서 진행

## 1. 목적

향후 LLM API를 연결할 때 단순한 결론이나 DB 값의 나열을 넘어서, 다음 질문에 연구자가
답을 얻을 수 있도록 한다.

- 어떤 데이터 소스를 어떤 질의로 조회했는가?
- 어떤 레코드가 어느 주장·가설·후보와 연결됐는가?
- 지지 근거와 반대 근거 중 무엇이 의사결정에 영향을 줬는가?
- 각 에이전트가 무엇을 전달·비판·수정했으며, 결과가 어떻게 바뀌었는가?
- 최종 결론의 불확실성, 누락된 근거, 대안은 무엇인가?

이 기능은 모델의 비공개 내부 추론(chain-of-thought)을 저장하는 기능이 아니다. 저장 대상은
입력 근거와 산출물에서 재검증 가능한 **구조화된 근거·판정 설명**이다.

## 2. 현재 구현과 추가 요구사항

| 구분 | 현재 구현 | LLM 연결 후 추가 요구사항 |
|---|---|---|
| 데이터 출처 | source, external ID, URL, 캐시·fallback·실패 상태 | 실제 질의, 조회 시각, 데이터 버전, 선택·제외 사유 |
| 주장–근거 연결 | evidence ID와 artifact ID | 주장별 지지·반박·맥락 근거 및 직접 사실/모델 해석 구분 |
| 에이전트 전달 | sender, receiver, 요약, 판정, 근거 ID | 요청·응답 요약, 쟁점, 미해결 질문, 수정 전후 차이 |
| 감사 | 검증 항목, finding, rationale, 환류 대상 | LLM 우려사항과 규칙 판정의 책임 분리, 의견 충돌 기록 |
| 모델 실행 | 모델명, 토큰, 비용 추정, 입력·출력 artifact | 응답 지연, 실패·fallback, 설명 레코드가 참조한 artifact hash |
| 사용자 화면 | 단계별 근거·판정 요약과 도구 호출 내역 | 연구자용 종합 서술, 결정 요인, 반대 근거, 불확실성, 대안 |

## 3. 실행별 필수 기록

### 3.1 데이터 수집 기록

- source name, adapter, endpoint class, dataset/API version
- 정규화된 검색 질의와 대상 질환·표적·후보 식별자
- 조회 시각, live/cache/local/fixture/fallback 구분
- 성공·빈 결과·미설정·시간 초과·오류 상태와 실패 유형
- 반환 레코드 수와 실제 채택된 evidence ID
- 해당 소스를 호출하거나 생략한 결정론적 라우팅 사유

### 3.2 근거 연결 기록

- claim ID → evidence ID 연결
- evidence 역할: support, counter, context
- 사실 수준: source fact, tool result, model interpretation
- 식별자 정규화 결과와 source version
- 인용 가능한 URL·외부 ID 및 원문을 복제하지 않는 excerpt hash
- 근거가 없는 주장에는 `unsupported`를 명시하고 결론 근거로 승격 금지

#### 3.2.1 허용 범위 내 초록·근거 문장 발췌

LLM 연결 여부와 무관하게, 원 출처의 라이선스·이용약관·접근권한이 허용하는 범위에서
초록 또는 결론을 직접 뒷받침하는 짧은 근거 문장을 별도로 조회·저장한다. LLM은 저장된
발췌를 설명에 연결할 수 있지만, 원문을 조회하지 않았거나 저장 계약을 충족하지 않은 상태에서
발췌를 생성·추정해서는 안 된다.

각 발췌 레코드는 최소한 다음 필드를 가진다.

- `excerpt_text`: 필요한 최소 길이의 짧은 원문 발췌
- `excerpt_type`: abstract, result sentence, caption, database description 등 원문 유형
- `source_location`: 초록, 섹션명, 문단, 페이지, 표·그림 번호 또는 API JSON 경로
- `source_url`, `external_id`, PMID/PMCID/DOI 등 안정 식별자
- `retrieved_at`, source/API version, live/cache 구분과 실제 조회 endpoint class
- 정규화 전 원문 바이트 또는 허용된 원문 조각의 SHA-256과 해시 대상·정규화 규칙
- 라이선스·접근 등급과 저장 허용 범위; 전문 저장이 금지된 출처는 URL·위치·해시만 보존
- claim/evidence/artifact ID 연결과 support/counter/context 역할
- 번역·요약이 있으면 원문 발췌와 분리하고 모델·버전·생성 시각을 별도 provenance로 기록

발췌 수집 실패, 유료벽, 로봇 차단, 라이선스 제한, 초록 부재는 서로 다른 상태로 기록한다.
이 경우 UI는 `원문 발췌 미기록`만 표시하지 않고 가능한 범위에서 미기록 사유를 함께 표시한다.
인용 길이는 출처 정책과 저작권 제한을 준수하고, 동일 원문의 과도한 누적 저장을 금지한다.

현재 PubMed·Europe PMC에는 위 원칙의 첫 구현이 적용되어 있다. 전체 초록은 저장하지 않고,
질환–표적 직접 연결 구문만 최대 320자로 보존하며 위치·content hash·excerpt hash를 함께
기록한다. 관련 구문이 없거나 초록 조회가 실패하면 제목을 지지 evidence로 승격하지 않는다.

중요하게, **LLM API 키를 연결하는 것만으로 다른 출처의 초록이나 원문 발췌가 자동으로
생기지는 않는다.** 출처별 원문 조회 어댑터, 저장 허용 정책, 위치 추출기, 해시 생성기와
무결성 검증을 별도로 구현하고 통과시켜야 한다.

### 3.3 LLM 판정 설명

LLM은 자유로운 내부 사고과정 대신 다음 구조를 출력한다.

- `conclusion`: 이번 단계의 결론
- `decision_factors`: 결정에 실제 사용된 요인과 evidence ID
- `counter_evidence`: 반대 근거와 영향
- `alternatives_considered`: 검토했지만 선택하지 않은 대안
- `uncertainties`: 데이터 부족, 충돌, OOD, 조회 실패
- `assumptions`: 명시적 가정
- `recommended_next_action`: 다음 에이전트 또는 연구자에게 전달할 작업
- `confidence_basis`: 성공확률이 아닌 근거 충족도와 그 산출 근거

### 3.4 에이전트 상호작용

- sender, receiver, agent role, model/provider
- 질문·업무지시 요약과 입력 artifact ID/hash
- 응답·비판 요약과 출력 artifact ID/hash
- 동의·충돌·보류 쟁점
- REVISE/REPLAN 전후 변경 필드와 변경 근거
- 해결되지 않은 질문과 Human Gate 전달 사유

## 4. 에이전트별 설명 범위

- **MA1**: 질환–표적 후보가 나온 출처, 기전 가설, 반증 근거, 대체 가설, 문헌 한계
- **G0/MA3**: 가설별 통과·실패 검증, 근거 추적성, 반증 가능성, 환류 이유
- **Planner**: 선택 작업·도구·모델·예산·중단 조건과 선택 근거
- **MA2**: 단일 탐색 경로 선택 근거, 후보별 생성/검색·화학·도킹·ADMET·정제 과정,
  금지·규제 검사, 순위 기여 요인
- **G1/MA3**: 후보 값과 tool trace의 일치성, 정책 위반, 모델·도구 적합성, 수정 범위
- **Human Gate**: 선택 가능한 후보, 핵심 근거·위험·불확실성, 보류 시 필요한 추가 자료

## 5. 안전·무결성 원칙

1. 원문 프롬프트, API 키, 개인정보 및 모델의 비공개 내부 추론은 저장하지 않는다.
2. LLM은 docking, ADMET, IC50 등 도구 계산값과 provenance를 생성하거나 덮어쓰지 못한다.
3. 모든 결정 요인은 실제 존재하는 evidence/artifact/tool trace ID를 참조해야 한다.
4. 참조가 없거나 artifact와 불일치하는 설명은 저장 전 검증에서 거부한다.
5. 규칙 엔진의 판정과 LLM의 해석·우려사항을 화면에서 명확히 구분한다.
6. fixture, cache, live source를 혼합 표기하지 않고 각 근거에 provenance를 표시한다.
7. 설명 레코드는 artifact hash와 함께 감사 원장에 결속한다.

## 6. 구현 순서

1. `ExplanationRecord` 구조화 스키마와 버전 정의
2. MA1·Planner·MA3 LLM 응답을 narrative overlay로 제한
3. evidence/artifact/tool trace 참조 무결성 검증기 추가
4. SQLite 저장 및 감사 원장 hash 결속
5. `/api/runs/{run_id}/explanations` 읽기 전용 API 추가
6. 화면에 `수집 데이터`, `결정 요인`, `반대 근거·불확실성`, `에이전트 상호작용` 구성
7. 정상·근거누락·출처충돌·조회실패·REVISE/REPLAN 회귀 테스트 추가
8. 허용 출처별 초록·근거 문장 수집기, 위치 추출기, SHA-256 및 라이선스 정책 검사 추가

## 7. 완료 기준

- 연구자가 결론마다 사용된 근거 ID와 원 출처를 화면에서 추적 가능
- 동일 artifact로 설명을 다시 생성했을 때 핵심 결정 요인이 재현 가능
- 출처 충돌·조회 실패·fixture fallback을 숨기지 않고 불확실성으로 표시
- 에이전트 간 전달 및 수정 전후 차이를 실행 타임라인에서 확인 가능
- LLM 설명이 실제 artifact와 다르면 저장·PASS 진행을 차단
- LLM을 끈 Mock 실행에서도 동일한 설명 스키마와 UI 계약 유지
- 발췌가 표시되면 원문 위치·안정 식별자·조회 시각·해시·허용 범위를 함께 추적 가능
- 원문 조회가 없거나 정책상 저장할 수 없는 경우 LLM이 발췌를 생성하지 않고 사유를 명시
