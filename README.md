# ece_30801_project
Phase 1 -> CLI

## API Keys

- Purdue: set `GEN_AI_STUDIO_API_KEY`
- Hugging Face: set `HF_TOKEN`

Usage options:
- Pass tokens explicitly to clients (tests do this):
  - `PurdueClient(max_requests=3, token="...")`
  - `HFClient(max_requests=3, token="...")`
- Or rely on environment variables:
  - `export GEN_AI_STUDIO_API_KEY=...`
  - `export HF_TOKEN=...`

Local setup:
- Copy `.env.example` to `.env` and fill in values if you use a loader like `python-dotenv` locally.
- `.env` is ignored by git.
