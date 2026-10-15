import numpy as np
import pytest
import yaml
from PIL import Image


@pytest.fixture
def frames(tmp_path):
    def write(relative, values, shape=(24, 28, 3)):
        directory = tmp_path / relative
        directory.mkdir(parents=True, exist_ok=True)
        for i, value in enumerate(values):
            array = (
                np.full(shape, value, dtype=np.uint8) if np.isscalar(value) else value
            )
            Image.fromarray(array).save(directory / f"frame{i}.png")
        return directory

    return write


@pytest.fixture
def config_path(tmp_path):
    def write(models, datasets, **options):
        data = {
            "version": 1,
            "models": {k: {"pred_root": str(v)} for k, v in models.items()},
            "datasets": datasets,
            "metrics": ["psnr", "ssim"],
            "runtime": {"device": "cpu"},
            **options,
        }
        path = tmp_path / "eval.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return write
