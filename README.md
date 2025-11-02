# ECE 30801 Project - Trustworthy Model Registry

Phase 1 → CLI | Phase 2 → REST API

## Phase 1: CLI

Evaluates trustworthy pre-trained model reuse with 8 metrics:
- Ramp-Up Time, License, Size, Availability
- Code Quality, Dataset Quality, Performance Claims, Bus Factor

## API Keys

**Purdue GenAI Studio API:**

**HuggingFace API:**

**Usage options:**
- Pass tokens explicitly to clients (tests do this):
  - `PurdueClient(max_requests=3, token="...")`
  - `HFClient(max_requests=3, token="...")`
- Or rely on environment variables:
  - `export GEN_AI_STUDIO_API_KEY=...`
  - `export HF_TOKEN=...`

**Local setup:**
- Copy `.env.example` to `.env` and fill in values if using `python-dotenv` locally
- `.env` is ignored by git

## Phase 2: REST API (In Progress)

REST API for model registry with AWS deployment (API Gateway, Lambda, DynamoDB, S3)

### Quick Start


### API Endpoints

- `GET /api/v1/health` - Health check
- `GET /api/v1/models` - List models
- `POST /api/v1/models/upload` - Upload model
- `POST /api/v1/models/ingest` - Ingest HuggingFace model
- `GET /api/v1/models/{model_id}` - Get model
- `DELETE /api/v1/models/{model_id}` - Delete model
- `POST /api/v1/reset` - Reset registry

### Deploy to AWS


## Project Structure


## Team

- Salaheldin Aboueitta
- Hussein Hamouda
- Eren Ulke
- Felix Wu
