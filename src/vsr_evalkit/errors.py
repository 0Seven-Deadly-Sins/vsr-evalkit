class EvaluationError(RuntimeError):
    """Actionable input, dependency or evaluation failure."""


class NotApplicable(EvaluationError):
    """The metric has no mathematically defined value for this sample."""
