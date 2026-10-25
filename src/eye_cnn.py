"""Open/closed eye CNN: model, inference wrapper, and a robust image-only trainer.

Train (from the project root):
    python -m src.eye_cnn --data data/eyes                 # folders named open / closed
    python -m src.eye_cnn --data PATH_TO_MRL               # raw MRL dataset also works
    python -m src.eye_cnn --data data/eyes --epochs 40 --self-repeat 3

Model contract (used by the whole project):
    input  : (N, 64, 64, 1) raw grayscale 0-255 crops (normalised inside the model)
    output : P(open) in [0, 1]   ->   closed probability = 1 - output
Right-eye crops are mirrored before inference, and training includes left-right flips.
"""
import argparse
import random
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import Config

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# ============================================================================ model
def build_model(size: int = 64):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input((size, size, 1))
    x = layers.RandomRotation(0.06)(inp)                    # train-time only
    x = layers.RandomTranslation(0.08, 0.08)(x)
    # per-image standardisation: removes brightness / contrast differences between cameras
    x = layers.LayerNormalization(axis=[1, 2, 3], center=False, scale=False, epsilon=1e-6)(x)
    for filters in (32, 64, 128):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.SpatialDropout2D(0.1)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)          # P(open)
    model = models.Model(inp, out, name="eye_state_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                  metrics=["accuracy"])
    return model


class EyeStateModel:
    """Inference wrapper used by the pipeline."""

    def __init__(self, path):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(str(path), compile=False)
        self.size = int(self.model.input_shape[1])

    def closed_probs(self, crops: List[np.ndarray]) -> List[float]:
        """P(closed) for every crop (right-eye crops must already be mirrored)."""
        x = np.stack(crops).astype("float32")[..., None]
        p_open = self.model(x, training=False).numpy().reshape(-1)
        return [float(v) for v in 1.0 - p_open]

    def closed_prob(self, crops: List[np.ndarray]) -> float:
        """Mean P(closed) over the crops (kept for backward compatibility)."""
        return float(np.mean(self.closed_probs(crops)))


def load_eye_model(path=None) -> Optional[EyeStateModel]:
    path = Path(path or Config().MODEL_PATH)
    if not path.exists():
        return None
    try:
        return EyeStateModel(path)
    except Exception as exc:                 # tensorflow missing or bad file
        print(f"[eye_cnn] could not load {path}: {exc}")
        return None


# ============================================================================ data
def label_and_group(p: Path):
    """Return (label, group): label 1 = open, 0 = closed; group keeps one person together."""
    parts = p.stem.split("_")
    if len(parts) >= 8 and parts[4] in ("0", "1") and re.fullmatch(r"s\d+", parts[0].lower()):
        return int(parts[4]), parts[0].lower()                       # MRL: group = person
    if p.name.startswith("self_") and len(parts) >= 4:               # own-camera frames
        try:
            lab = 1 if parts[2] == "open" else 0
            return lab, f"{parts[1]}_{parts[2]}_{int(parts[3]) // 60}"   # block of 60 frames
        except ValueError:
            pass
    for parent in list(p.parents)[:3]:                               # nearest folder name wins
        name = parent.name.lower()
        if "close" in name:
            return 0, None
        if "open" in name:
            return 1, None
    return None, None


def scan(roots, limit: int, seed: int = 42):
    paths, labels, groups, skipped = [], [], [], 0
    for root in roots:
        for p in Path(root).rglob("*"):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lab, grp = label_and_group(p)
            if lab is None:
                skipped += 1
                continue
            paths.append(str(p))
            labels.append(lab)
            groups.append(grp or f"img_{len(paths)}")
    if not paths:
        raise SystemExit("No labelled images found. Use folders called open/closed, or the MRL dataset.")
    labels = np.array(labels)
    rng, keep = random.Random(seed), []
    for c in (0, 1):                                  # cap each class so neither dominates
        idx = [i for i in range(len(paths)) if labels[i] == c]
        rng.shuffle(idx)
        keep += idx[:limit]
    keep.sort()
    print(f"found {len(paths)} images ({skipped} skipped, no label)  ->  using {len(keep)}")
    return [paths[i] for i in keep], labels[keep], np.array([groups[i] for i in keep])


def split(paths, labels, groups, seed: int = 42):
    """70/15/15 split that never puts one person / frame block in two splits."""
    from sklearn.model_selection import GroupShuffleSplit

    paths = np.array(paths)
    tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(paths, labels, groups))
    va, te = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(paths[tmp], labels[tmp], groups[tmp]))
    va, te = tmp[va], tmp[te]
    out = [(list(paths[i]), labels[i].astype("float32")) for i in (tr, va, te)]
    for name, (p, y) in zip(("train", "val", "test"), out):
        print(f"{name:5s} {len(p):6d}  closed={int((y == 0).sum())}  open={int((y == 1).sum())}")
    return out


def make_ds(paths, labels, batch: int, size: int = 64, train: bool = False, degrade: bool = False):
    import tensorflow as tf

    def load(path, y):
        img = tf.image.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        img.set_shape((size, size, 1))
        return tf.cast(img, tf.float32), y

    def augment(img, y):                                       # simulate a normal webcam
        scale = tf.random.uniform([], 0.75, 1.30)              # framing: tighter / looser crop
        n = tf.cast(size * scale, tf.int32)
        img = tf.image.resize_with_crop_or_pad(tf.image.resize(img, tf.stack([n, n])), size, size)
        s = tf.random.uniform([], 20, size + 1, dtype=tf.int32)          # lose resolution
        img = tf.image.resize(tf.image.resize(img, tf.stack([s, s])), (size, size))
        gamma = tf.random.uniform([], 0.55, 1.7)
        img = 255.0 * tf.pow(tf.clip_by_value(img, 0.0, 255.0) / 255.0, gamma)
        img = tf.image.random_contrast(img, 0.5, 1.5)
        img = tf.image.random_brightness(img, 45.0)
        img += tf.random.normal(tf.shape(img), 0.0, tf.random.uniform([], 0.0, 14.0))
        img = tf.image.random_flip_left_right(tf.clip_by_value(img, 0.0, 255.0))
        img.set_shape((size, size, 1))
        return img, y

    def degrade_fixed(img, y):                                 # deterministic "bad webcam" test
        img = tf.image.resize(tf.image.resize(img, (28, 28)), (size, size))
        img = 255.0 * tf.pow(tf.clip_by_value(img, 0.0, 255.0) / 255.0, 1.4)
        noise = tf.random.stateless_normal(tf.shape(img), (7, 7), 0.0, 8.0)
        return tf.clip_by_value(img * 0.7 + 10.0 + noise, 0.0, 255.0), y

    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels, "float32")))
    if train:
        ds = ds.shuffle(min(len(paths), 20000), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    if degrade:
        ds = ds.map(degrade_fixed, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ============================================================================ train
def report(model, ds, y, title: str, out_png=None) -> float:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    p_open = model.predict(ds, verbose=0).reshape(-1)
    pred = (p_open >= 0.5).astype(int)
    y = y.astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    closed_recall = cm[0, 0] / max(cm[0].sum(), 1)
    open_recall = cm[1, 1] / max(cm[1].sum(), 1)
    print(f"\n=== {title} ===")
    print(classification_report(y, pred, labels=[0, 1], target_names=["closed", "open"], digits=4, zero_division=0))
    try:
        print(f"ROC-AUC {roc_auc_score(y, p_open):.4f}", end="   ")
    except ValueError:
        pass
    print(f"CLOSED-eye recall {closed_recall:.3f}   open-eye recall {open_recall:.3f}")
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["closed", "open"]).plot(ax=ax, colorbar=False)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    return float(closed_recall)


def train(roots, out_path=None, epochs: int = 40, batch: int = 64, size: int = 64,
          limit: int = 25000, self_repeat: int = 3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    cfg = Config()
    out_path = Path(out_path or cfg.MODEL_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)

    paths, labels, groups = scan(roots, limit)
    (ptr, ytr), (pva, yva), (pte, yte) = split(paths, labels, groups)

    if self_repeat > 1:                           # emphasise your own webcam samples
        extra = [(p, y) for p, y in zip(ptr, ytr) if Path(p).name.startswith("self_")]
        if extra:
            ptr = list(ptr) + [p for p, _ in extra] * (self_repeat - 1)
            ytr = np.concatenate([ytr, np.array([y for _, y in extra] * (self_repeat - 1), dtype=ytr.dtype)])
            print(f"own-camera samples in train: {len(extra)} (x{self_repeat})")

    cw = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=ytr)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    ds_tr = make_ds(ptr, ytr, batch, size, train=True)
    ds_va = make_ds(pva, yva, batch, size)
    ds_te = make_ds(pte, yte, batch, size)
    ds_te_bad = make_ds(pte, yte, batch, size, degrade=True)

    model = build_model(size)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(str(out_path), monitor="val_loss", save_best_only=True),
    ]
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, callbacks=cbs, class_weight=class_weight)
    model.save(out_path)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axs, ("loss", "accuracy")):
        ax.plot(hist.history[key], label="train")
        ax.plot(hist.history["val_" + key], label="val")
        ax.set_title(key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)

    r1 = report(model, ds_te, yte, "TEST - clean images", cfg.OUTPUT_DIR / "confusion_matrix.png")
    r2 = report(model, ds_te_bad, yte, "TEST - webcam-degraded images", cfg.OUTPUT_DIR / "confusion_matrix_degraded.png")
    print(f"\nmodel saved to {out_path}")
    if min(r1, r2) < 0.90:
        print("WARNING: closed-eye recall is below 90%. Add more closed-eye images, or record your own samples "
              "(python -m src.collect --webcam) and train again.")
    else:
        print("Next: python scripts/check_model.py  and close your eyes in front of the camera to confirm.")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", "--root", dest="data", nargs="+", default=["data/eyes"],
                    help="folder(s) with open/closed images, or the raw MRL dataset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=25000, help="max images per class")
    ap.add_argument("--self-repeat", type=int, default=3, help="repeat own-camera (self_*) images in training")
    a = ap.parse_args()
    train(a.data, a.out, a.epochs, a.batch, limit=a.limit, self_repeat=a.self_repeat)


if __name__ == "__main__":
    main()