# ece_30801_project
Phase 1 -> CLI

COMMENTING PRACTICES:
def function_with_comment_below():
    """
    Description of function
    Compute the metric score from parsed inputs.

    Parameters
    ----------
    inputs : dict[str, Any]
        Parsed inputs required by the metric.
    **kwargs : Any
        Optional per-metric tuning parameters.

    Returns
    -------
    output : float
        A score between 0.0 and 1.0.

    Raises
    ------
    RuntimeError
        If required inputs are missing or invalid.
    Exception
        Reason
    """

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
