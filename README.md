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