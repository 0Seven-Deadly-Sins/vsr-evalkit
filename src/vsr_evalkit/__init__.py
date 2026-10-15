"""VSR evaluation, without importing any restoration model."""

__version__ = "0.1.0"


def evaluate(config, models, datasets, **kwargs):
    """Evaluate registered model/dataset IDs. See runner.evaluate for options."""
    from .runner import evaluate as run

    return run(config, models, datasets, **kwargs)
