# DrugPilot 오프라인 평가 프레임워크

## 목적

이 프레임워크는 현재의 deterministic 방식과 향후 LLM/Hermes 방식을 동일한 고정 입력과 정답 정의로 비교한다. 평가 코드는 실행 그래프, 외부 데이터 어댑터, `.env`와 분리되어 있으며 API 키 없이 이미 기록된 결과를 채점할 수 있다.

현재 포함된 `data/evaluation/demo_v1`은 연결과 지표 계산을 검증하기 위한 **비운영·예시 데이터셋**이다. 정답과 universe 크기는 과학적으로 검증된 gold standard가 아니므로 모델 성능이나 임상적 유효성을 주장하는 데 사용할 수 없다.

## 검색 오염 방지 경계

```text
benchmark.json
  ├─ public query ───────────────▶ deterministic / LLM / Hermes executor
  │                                  │
  │                                  ▼
  │                              CasePrediction
  │                                  │
  └─ scorer-only labels ─────────────┴──▶ metric scorer
```

- 실행기에는 `EvaluationQuery(case_id, disease, research_goal, attributes)`만 전달된다.
- 관련 표적, known hit, 기대 분기와 완전성 표시는 `BenchmarkLabels`에 남고 실행 입력에 포함되지 않는다.
- 각 결과는 label-free 입력의 SHA-256인 `input_digest`를 기록해야 한다.
- 전체 benchmark digest와 case label digest가 retrieval artifact provenance에 나타나면 채점을 거부한다.
- `label_fields_received` 또는 `benchmark_payload_accessed` attestation이 설정되어도 채점을 거부한다.
- benchmark와 실행 결과는 모두 `frozen` Pydantic 모델이며 benchmark checksum으로 버전을 고정한다.

이 검사는 우발적 오염을 탐지하는 감사 경계다. 악의적인 코드의 파일 접근을 막는 보안 sandbox는 아니다. 운영 평가에서는 실행기와 scorer를 별도 프로세스 또는 컨테이너로 나누고 benchmark label 파일을 실행기 파일시스템에 마운트하지 않아야 한다.

## 지표

- `target_recall_at_k`: 정답 표적 중 상위 K에 회수된 비율
- `candidate_recall_at_k` / `known_hit_recovery_at_k`: known candidate hit 회수율
- `target_precision_at_k`, `candidate_precision_at_k`: 완전한 정답셋에서만 계산하며 분모는 K
- `target_enrichment_at_k`, `candidate_enrichment_at_k`: `Precision@K / 정답 prevalence`; 완전한 정답과 universe 크기가 모두 있을 때만 계산
- `target_false_exclusion_rate`, `candidate_false_exclusion_rate`: 관련 정답을 명시적으로 제외한 비율
- `target_branch_coverage`: 기대한 표적 분기 중 실제 시도한 비율
- `source_failure_rate`, `tool_failure_rate`, `source_tool_failure_rate`: `skipped`를 제외한 호출 중 failed/timeout/unavailable 비율
- `runtime_ms`, `runtime_ms_total`: case별 실행시간과 전체 시간

case별 점수와 전체 macro 평균을 함께 저장한다. 완전하지 않은 라벨에 대한 Precision/Enrichment는 0으로 대체하지 않고 `null`과 사유를 남긴다.

## 사용 인터페이스

실행기를 붙일 때는 label-free protocol만 구현한다.

```python
from drugpilot.evaluation import BenchmarkRunner, EvaluationMode, load_benchmark

class DeterministicAdapter:
    mode = EvaluationMode.DETERMINISTIC

    def predict(self, query):
        # query에는 benchmark labels가 없다.
        return recorded_case_prediction

benchmark = load_benchmark(
    "data/evaluation/demo_v1/benchmark.json",
    checksum_path="data/evaluation/demo_v1/benchmark.sha256",
)
result = BenchmarkRunner().execute(benchmark, DeterministicAdapter())
```

향후 LLM/Hermes 실행은 각각 `mode="llm"`, `mode="hermes"`인 `EvaluationRun`을 저장한다. API 키가 없는 scorer 환경에서는 다음처럼 JSON 산출물만 비교한다.

```powershell
.venv\Scripts\python.exe scripts\evaluate_recorded_runs.py `
  --benchmark data\evaluation\demo_v1\benchmark.json `
  --checksum data\evaluation\demo_v1\benchmark.sha256 `
  --run results\deterministic.json `
  --run results\llm.json `
  --run results\hermes.json
```

`BenchmarkRunner.compare()`는 deterministic 결과를 기준으로 동일 metric/K의 차이를 계산한다. 숫자 지표가 없는 narrative 차이는 이 평가 객체의 범위 밖이며 별도의 blinded human rubric이 필요하다.

## 저장된 DrugPilot 실행을 평가 형식으로 변환

웹/API에서 내려받은 `RunSnapshot` JSON은 읽기 전용 adapter로 변환할 수 있다. adapter는 SQLite를 열거나 외부 API를 호출하지 않으며, benchmark 정답 대신 `EvaluationQuery`만 받는다. 각 snapshot은 해당 case와 질환명·연구목표·research depth가 일치해야 한다.

```powershell
.venv\Scripts\python.exe scripts\export_snapshot_evaluation.py `
  --benchmark data\evaluation\demo_v1\benchmark.json `
  --checksum data\evaluation\demo_v1\benchmark.sha256 `
  --snapshot demo-aml-001=exports\aml-run.json `
  --snapshot demo-tnbc-001=exports\tnbc-run.json `
  --mode deterministic `
  --output results\deterministic.json
```

변환 결과에는 다음이 보존된다.

- snapshot run ID, stage, runtime mode, 전체 snapshot digest
- MA1 표적 순위와 UniProt/Ensembl identifier
- 후보 이름, 내부 candidate ID, InChIKey, SMILES, ChEMBL/DrugCentral/PubChem identifier
- candidate/target/portfolio 원래 순위와 평가용 전역 순위
- 제외 상태와 reason code
- MA2 tool call의 원래 execution status, 시간, cache/fixture, 버전, 연결 후보
- public-source call trace의 상태와 granularity

`observed_records`와 `selection_decision` trace는 감사에는 남지만 실제 호출 실패율의 분모에는 포함하지 않는다. SQLite 파일을 직접 여는 기능은 의도적으로 제공하지 않는다. 실행 중인 WAL 데이터베이스에 잠금 또는 부수 파일을 만들 가능성을 없애기 위해 먼저 API/SQLite에서 `snapshot_json`을 파일로 export한 뒤 이 adapter에 전달한다.

## 운영 평가셋으로 전환하기 전 필요한 작업

1. 질환·표적·후보 정답을 독립적인 전문가 검토와 체계적 문헌 기준으로 확정한다.
2. 탐색에 사용한 DrugCentral/ChEMBL/PubMed snapshot과 시간적으로 격리된 평가 snapshot을 분리한다.
3. 후보 universe와 negative/unknown의 의미를 고정하고 exhaustive 여부를 case별로 명시한다.
4. 후보 식별자는 이름뿐 아니라 InChIKey/ChEMBL ID 등 안정적인 alias를 함께 기록한다.
5. benchmark 버전 변경 시 이전 파일을 덮어쓰지 않고 새 디렉터리와 checksum을 만든다.
6. API 장애 평가와 과학 성능 평가를 분리해 반복 실행하고 평균·분산을 기록한다.

## 현재 한계

- demo labels는 예시이며 production 승인되지 않았다.
- 후보 유사구조 회수는 아직 exact identifier/alias 일치만 사용한다.
- confidence calibration, AUROC/AUPRC, 시간 분할 검증, decoy benchmark는 아직 포함되지 않았다.
- LLM/Hermes adapter는 API 키 확보 후 별도로 구현해야 한다. 프레임워크는 지금도 그 산출물을 오프라인 비교할 수 있다.
