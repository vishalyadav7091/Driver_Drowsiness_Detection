import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.eye_cnn import build_model


def test_model_output_shape_and_range():
    m = build_model(64)
    out = m(np.random.randint(0, 255, (4, 64, 64, 1)).astype("float32"), training=False).numpy()
    assert out.shape == (4, 1) and ((out >= 0) & (out <= 1)).all()
