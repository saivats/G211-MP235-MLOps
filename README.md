# MP235 — Cold Storage Spoilage Prediction (Guwahati)

**Course:** SCSE3040 — MLOps Mini Project (Assignment, 10 marks)  
**Group:** G211 | **Batch:** 55  
**Faculty:** Dr. Saurabh Shrivastava  
**Problem ID:** MP235  
**Client scenario:** Guwahati Cold Chain Cooperative (fictional)  
**Status:** In progress — the checklist below is a plan, **not** a statement of completed work.

## 1. Problem overview

Build a reproducible MLOps pipeline to predict `spoilage_pct` (percentage of a stored lot spoiled) for potatoes, onions and apples. This is a **regression** task, evaluated by **mean absolute error (MAE)**. The full assignment includes data audit, versioning, cleaning, MLflow experiment tracking and registry, prediction API, Docker, GitHub Actions, Kubernetes, structured logging, drift, retraining and final holdout predictions.

**Important:** Follow the unique group-provided `PROBLEM.md`, `STUDENT_HANDBOOK.md` and starter `findings.yaml` locally. Do not publish the original dataset in ordinary Git commits.

## 2. Group G211 — distinct primary responsibilities

| Member | Enrollment number | Role | Primary ownership |
|---|---|---|---|
| **Saivats** | `S24CSEU1436` | **ML & Experiment Lead** | Reproducible regression training, 3+ MLflow experiment runs, data SHA-256 logging, MAE evaluation, model registry, `champion` alias, v1/v2 comparison and final holdout predictions |
| **Pradyumna Singh** | `S24CSEU1400` | **Data Engineering & DVC Lead** | Preserve original data hashes, DVC setup, checks for all seven data-quality problems, cleaning pipeline, validation tests, data findings |
| **Deepanshu Bhadana** | `S24CSEU0839` | **API & Container Lead** | FastAPI `/predict` and `/health`, request validation and HTTP 422 tests, model loading, Dockerfile and container verification |
| **Daksh Kasana** | `S24CSEU2714` | **DevOps & Monitoring Lead** | GitHub Actions CI, Kubernetes deployment/service, JSON stdout logging, PSI drift analysis, chamber-wise monitoring and retraining workflow integration |

**Shared responsibility:** Every member reviews integrations, understands the complete pipeline for the individual viva, makes their **own genuine GitHub commits**, and submits the Microsoft Form separately. The table assigns planned ownership; it does not assert completed contributions. The group may revise these assignments collaboratively.

### Working areas by owner

- **Saivats:** `src/<package>/train*.py`, model evaluation, MLflow experiment outputs, `models/model.joblib`, `predictions/`.
- **Pradyumna:** `.dvc/`, `data/raw/*.csv.dvc`, `src/<package>/data*.py`, data tests, data sections of `reports/findings.yaml`.
- **Deepanshu:** `app/main.py`, API tests, `Dockerfile`, `.dockerignore`.
- **Daksh:** `.github/workflows/ci.yml`, `k8s/`, monitoring/retraining modules and tests, operational report fields.

Files that span roles (for example `requirements.txt`, `params.yaml`, `reports/findings.yaml`) must be reviewed together.

## 3. Official timeline and assessment

| Stage | Dates (2026) | Expected outcome |
|---|---|---|
| Stage 1 | 18–20 November | Public GitHub repository, DVC, initial data audit |
| Stage 2 | 21–23 November | Cleaning, MLflow and model registry, tests |
| Stage 3 | 24–26 November | FastAPI, Dockerfile, green GitHub Actions, Kubernetes, logging |
| Stage 4 | 27–29 November | Drift, retraining, holdout predictions, findings |

**Submission deadline:** **Sunday, 29 November 2026 at 11:59 PM IST**.  
**Grace period:** Through Friday, 4 December 2026 at 11:59 PM IST (reduced timeliness marks).  
**Individual viva:** Week beginning Monday, 30 November 2026.

**Rubric:** Timeliness 3 marks; project completeness 4; individual viva 3. The marking tool also evaluates individual contributions. Every member submits the **same repository URL** through [the individual submission form](https://forms.cloud.microsoft/r/9ZPB1ZzcAK).

Stages 1 and 2 are earlier milestones. Complete any outstanding tasks with **real present-day work and commit timestamps**. Never backdate or invent commit history.

## 4. Daily task and commit schedule (24–29 November)

This is our suggested recovery/implementation plan. **The faculty does not require one commit per day**; it requires authentic work from each member. Suggested commit messages are examples to use **only after the corresponding code or documentation is actually changed and checked**.

### Tuesday, 24 November — foundation and pending Stage 1/2 tasks

| Member | Task | Example commit message |
|---|---|---|
| Saivats | Plan feature preprocessing; define `params.yaml` model settings and reproducible training interface | `feat(model): add configurable training entry point` |
| Pradyumna | Initialize DVC; record untouched CSV hashes; implement preliminary checks for the seven anomalies | `feat(data): initialize DVC and data audit` |
| Deepanshu | Define eight-field API schema and 422 constraints; implement `/health` and basic TestClient tests | `feat(api): define schemas and health endpoint` |
| Daksh | Add pytest-on-push CI and first Kubernetes Deployment/Service manifests | `ci(ops): add tests workflow and k8s manifests` |

### Wednesday, 25 November — baseline components

| Member | Task | Example commit message |
|---|---|---|
| Saivats | Train baseline v1 on cleaned train data; log MAE and original data SHA-256 to MLflow SQLite | `feat(model): train baseline with MLflow tracking` |
| Pradyumna | Implement cleaning functions for confirmed problems; add regression tests and data evidence | `fix(data): clean verified anomalies` |
| Deepanshu | Connect FastAPI `/predict` to real model loading, add request/response tests | `feat(api): implement model prediction endpoint` |
| Daksh | Implement JSON-per-prediction stdout logs with API owner; start PSI utility and tests | `feat(ops): add structured logging and psi utility` |

### Thursday, 26 November — tracking, packaging, integration gate

| Member | Task | Example commit message |
|---|---|---|
| Saivats | Complete at least three MLflow runs, compare MAE, register v1 and set `champion` | `feat(model): register tracked champion model` |
| Pradyumna | Validate data leakage exclusion, cleaning tests, original hashes and DVC metadata | `test(data): cover raw data integrity` |
| Deepanshu | Add Docker build using `python:3.12-slim`, expose 8000 and validate container if available | `build(api): containerize prediction service` |
| Daksh | Make CI green; verify three-replica manifests, exact names, routes and probes | `fix(ci): verify pipeline and health probes` |

### Friday, 27 November — monitoring and service checks

| Member | Task | Example commit message |
|---|---|---|
| Saivats | Evaluate v1 against `later.csv`, including MAE by chamber | `feat(model): report chamber-wise later mae` |
| Pradyumna | Verify cleaned data consistency and evidence for each of seven issue types | `docs(data): document cleaning evidence` |
| Deepanshu | Add API boundary tests and verify HTTP 422 and `/health` | `test(api): add boundary and health tests` |
| Daksh | Calculate 10-bin PSI for all six numeric predictors and identify maximum | `feat(monitor): calculate feature psi drift` |

### Saturday, 28 November — retraining and final integration

| Member | Task | Example commit message |
|---|---|---|
| Saivats | Train v2 on cleaned train + earliest 70% of later; evaluate latest 30%; apply 8% promotion rule | `feat(model): evaluate v2 promotion rule` |
| Pradyumna | Review measured findings and synchronize required `reports/findings.yaml` data section | `test(data): validate findings evidence` |
| Deepanshu | Verify exported champion model loads in API; run integration tests | `test(api): verify model integration` |
| Daksh | Connect drift report with retraining workflow, verify CI/Kubernetes end-to-end | `feat(ops): integrate monitoring and retraining` |

### Sunday, 29 November — release quality gate

| Member | Task | Example commit message |
|---|---|---|
| Saivats | Produce champion holdout predictions, validate exactly `record_id,prediction` for all 250 records | `feat(predictions): export holdout predictions` |
| Pradyumna | Finish data portion of `reports/findings.yaml`; verify DVC pointers and CSV hashes | `docs(findings): finalize measured data checks` |
| Deepanshu | Run full API/test suite, inspect container behavior and document verified commands | `test(api): complete service verification` |
| Daksh | Check final green GitHub Actions, monitoring, logs and Kubernetes manifests | `chore(ops): verify release checklist` |

**All members, 29 November:** Review the final repository and evidence; fill the submission form individually by 11:59 PM IST. Use truthful descriptions of personally completed work.

## 5. Expected repository structure

```text
README.md
requirements.txt                  # pinned dependencies
params.yaml                       # loaded via yaml.safe_load
src/<package>/__init__.py
src/<package>/*.py                 # data, training, evaluation, monitoring
tests/test_*.py                   # >=3 pytest tests, >=1 TestClient test
data/raw/train.csv.dvc            # raw CSV not in Git
data/raw/later.csv.dvc            # raw CSV not in Git
data/raw/holdout_features.csv
.dvc/config
mlflow.db                         # SQLite database with logged runs/registry
app/main.py
models/model.joblib               # deployed champion
Dockerfile
.dockerignore
.github/workflows/ci.yml
k8s/deployment.yaml
k8s/service.yaml
reports/findings.yaml            # measured values, no placeholders at submission
predictions/holdout_predictions.csv
```

This tree lists **required deliverables**, not files already present.

## 6. Project-specific acceptance criteria

- Inspect all seven anomaly types: `leaking_column`, `unit_mixup`, `sentinel_value`, `duplicate_rows`, `stuck_sensor`, `target_typos`, `spelling_variants`. Record true/false and exact details only when supported by real analysis.
- Preserve the exact original training CSV bytes; do **not** open and resave in Excel. DVC-track train/later without committing CSV contents directly.
- MLflow SQLite tracking at `sqlite:///mlflow.db`; three or more real runs, original `train.csv` `data_sha256` logged in each; logged MAE, registered v1 and v2, `champion` alias.
- `POST /predict` accepts **exactly** `storage_days`, `temp_set_c`, `humidity_pct`, `power_cuts_hours`, `bag_weight_kg`, `door_openings_per_day`, `crop`, `chamber`. Reject `storage_days > 300` and `bag_weight_kg < 20` with HTTP 422 using Pydantic validation; `GET /health` returns `{"status":"ok"}`.
- Docker image: `mp235-api:1.0`. Kubernetes Deployment: `mp235-api` with **3 replicas**, container port 8000. Service: `mp235-svc`, port 80 mapped to 8000. Readiness and liveness on `/health` with a later liveness start.
- JSON stdout prediction logs. Green GitHub Actions pytest workflow on every push.
- PSI with **10 bins** for every numeric predictor on cleaned train vs later; v1 MAE by chamber on later; v2 trained on cleaned train plus first **70%** of date-sorted later and evaluated on remaining **30%**. Promote only if v2 MAE is **at least 8% lower** than v1 on that check period.
- Final holdout file: `predictions/holdout_predictions.csv`, exactly `record_id,prediction` for all supplied records.

## 7. Collaboration guidelines

1. Work in topic branches such as `data/audit`, `model/training`, `api/service`, `ops/ci`. Use pull requests for review and integrate to `main`.
2. Every member must be added as a collaborator and commit from **their own account**. Never share credentials, manufacture contribution history, or imply another member did work that they did not.
3. Commit meaningful increments with tests/documentation. One daily commit is an **optional coordination goal**, not an official minimum.
4. Do not publish the group-specific supplied CSV files in regular Git history. Follow the faculty's DVC instructions and project sharing restrictions.
5. Complete this coursework yourselves. The handbook permits AI for explanation/debugging, not as a substitute for implementing and understanding your work; be prepared to explain and modify every contribution in the viva.
6. Never invent experiment metrics, MLflow run IDs, CI status or `findings.yaml` values.

## 8. Submission

Each of the four members must fill [the individual Microsoft Form](https://forms.cloud.microsoft/r/9ZPB1ZzcAK) before **29 November 2026, 11:59 PM IST** with this same public repository URL:

**https://github.com/saivats/G211-MP235-MLOps**

**Current repo status:** Planning/coordination initialized; data and implementation deliverables still require member-authored work.

## PSI drift monitoring

After the group-provided `data/raw/train.csv` and `data/raw/later.csv` are available and Pradyumna's `mp235.cleaning.clean_data` module is integrated, run the monitor from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m mp235.monitor --train data/raw/train.csv --later data/raw/later.csv
```

The command cleans the training reference with the shared cleaning pipeline, then prints JSON containing 10-bin PSI for the six configured numeric predictors, the highest-PSI feature, and reference/current distribution summaries. PSI excludes missing and non-finite observations; their counts are included in each summary. Do not copy synthetic test metrics into `reports/findings.yaml`.
