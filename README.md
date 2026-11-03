# Trustworthy Model Registry

A registry for pre-trained machine-learning models that scores each model on trust metrics before
anyone reuses it. Built by a team of four for Purdue ECE 30861 (Software Engineering), Fall 2025.

- **Phase 1** is a command-line tool that fetches Hugging Face models, scores them on 8 metrics in
  parallel, and writes one NDJSON record per model. Its original repository is
  [ECE30861](https://github.com/Salah-elsayed-2005/ECE30861); the code also lives in `src/` here.
- **Phase 2** (this repository) puts the scoring behind an OpenAPI REST service on AWS, with JWT
  access control, artifact storage, lineage and a browser UI.

## Architecture

```
browser UI (frontend/) ─┐
CLI / course autograder ─┼─► API Gateway (HTTP API) ─► AWS Lambda: FastAPI + Mangum ─┬─► DynamoDB  metadata, users, audit log
                         │                              (src/lambda_handler.py)       └─► S3        model artifacts
```

Infrastructure is Terraform (`infra/terraform/`). Pushing to `main` runs
`.github/workflows/deploy.yml`, which builds the Lambda package and updates the function.

## Trust metrics

| Metric | Phase | What it measures | Class in `src/Metrics.py` |
| --- | --- | --- | --- |
| Ramp-up time | 1 | How quickly a new user can get started from the model card | `RampUpTime` |
| License | 1 | License clarity and compatibility | `LicenseMetric` |
| Size | 1 | Fit for Raspberry Pi, Jetson, desktop and cloud | `SizeMetric` |
| Dataset and code availability | 1 | Whether training data and code are linked | `AvailabilityMetric` |
| Performance claims | 1 | Evidence of benchmarks or evaluations | `PerformanceClaimsMetric` |
| Dataset quality | 1 | Health of the training data | `DatasetQuality` |
| Code quality | 1 | Health of the linked repository | `CodeQuality` |
| Bus factor | 1 | Knowledge concentration among maintainers | `BusFactorMetric` |
| Reproducibility | 2 | Whether the demo code runs as published | `ReproducibilityMetric` |
| Reviewedness | 2 | Share of code that arrived through reviewed pull requests | `ReviewednessMetric` |
| Treescore | 2 | Average score of parent models in the lineage | `TreescoreMetric` |

A weighted net score combines the Phase 1 metrics. Each metric reports its own latency.

## API

The Lambda serves `src/api/autograder_routes.py`, which implements the course OpenAPI contract
(`docs/openapi-spec.yaml`).

| Route | Purpose |
| --- | --- |
| `PUT /authenticate` | Exchange a username and password for a JWT |
| `POST /artifact/{type}` | Ingest a model, dataset or code artifact |
| `POST /artifacts` | List or query artifacts |
| `GET /artifact/byName/{name}`, `POST /artifact/byRegEx` | Search |
| `GET` / `PUT` / `DELETE /artifacts/{type}/{id}` | Read, update, delete |
| `GET /artifact/model/{id}/rate` | Trust scores for a model |
| `GET /artifact/model/{id}/lineage` | Parent-model graph |
| `GET /artifact/{type}/{id}/cost` | Download size cost |
| `POST /artifact/model/{id}/license-check` | License compatibility check |
| `GET /artifact/{type}/{id}/audit` | Audit trail |
| `DELETE /reset` | Reset the registry (admin token required) |
| `GET /health`, `GET /tracks` | Health and the implemented course track |

## Security

Clients authenticate with `PUT /authenticate` and send the returned token as a bearer token.
Tokens are signed JWTs (HS256). Passwords are stored as salted SHA-256 hashes in DynamoDB, and
resetting the registry requires an admin token.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GEN_AI_STUDIO_API_KEY and HF_TOKEN
pytest tests/ --cov=src         # unit and API tests
./run /absolute/path/to/URL_FILE   # Phase 1 scoring CLI
```

## Deploy

```bash
cd infra/terraform
terraform init
terraform apply                 # S3 bucket, DynamoDB table, Lambda, API Gateway, IAM
```

After that, each push to `main` redeploys the Lambda. The workflow needs AWS credentials stored
as repository secrets.

## Repository layout

```
src/              Python package: CLI, metrics, API (src/api/), Lambda entry point
tests/            pytest suite
infra/terraform/  AWS infrastructure
frontend/         static browser UI
examples/         reproducibility demo
docs/             OpenAPI spec and development notes
scripts/dev/      one-off debugging and deployment scripts kept for reference
src/index.ts, src/handlers/   early TypeScript prototype; the deployed API is the Python one
```

## Team

Salaheldin Aboueitta, Hussein Hamouda, Eren Ulke, Felix Wu.

## License

See `LICENSE`.
