import numpy as np

from src import video_pipeline
from src.config import Config


def test_pipeline_mirrors_right_eye_before_cnn(monkeypatch):
    landmarks = np.zeros((478, 2), dtype=np.float32)
    landmarks[video_pipeline.LEFT_EYE, 0] = 1
    landmarks[video_pipeline.RIGHT_EYE, 0] = 2

    class FakeMesh:
        def process(self, _frame):
            return landmarks

        def close(self):
            pass

    class FakeModel:
        crops = None

        def closed_prob(self, crops):
            self.crops = crops
            return 0.5

    def fake_crop(_gray, points, _size):
        if points[0, 0] == 1:
            return np.array([[1, 2], [3, 4]], dtype=np.uint8)
        return np.array([[5, 6], [7, 8]], dtype=np.uint8)

    def noop(*_args):
        pass

    monkeypatch.setattr(video_pipeline, "FaceMeshDetector", FakeMesh)
    monkeypatch.setattr(video_pipeline, "crop_eye", fake_crop)
    monkeypatch.setattr(video_pipeline, "head_pose", lambda *_args: None)
    monkeypatch.setattr(video_pipeline, "draw_eye", noop)
    monkeypatch.setattr(video_pipeline, "draw_hud", noop)

    model = FakeModel()
    pipeline = video_pipeline.DrowsinessPipeline(Config(CNN_EVERY=1), model)
    pipeline.process_frame(np.zeros((32, 32, 3), dtype=np.uint8), 0)

    assert len(model.crops) == 2
    np.testing.assert_array_equal(model.crops[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(model.crops[1], [[6, 5], [8, 7]])
