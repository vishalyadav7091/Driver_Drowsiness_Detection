"""Eye-state dataset: expects data/eyes/closed/*.png and data/eyes/open/*.png."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

CLASSES = ["closed", "open"]          # label 0 = closed, 1 = open
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(data_dir) -> Tuple[List[str], np.ndarray]:
    data_dir = Path(data_dir)
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        folder = data_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")
        files = [str(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXT]
        paths += files
        labels += [label] * len(files)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")
    return paths, np.array(labels, dtype=np.int32)


def split_70_15_15(paths, labels, seed: int = 42):
    """Stratified 70/15/15 split."""
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return (p_tr, y_tr), (p_va, y_va), (p_te, y_te)


def make_ds(paths, labels, size: int = 64, batch: int = 64, shuffle: bool = False):
    import tensorflow as tf

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        img.set_shape((size, size, 1))
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels)))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 10000), seed=42, reshuffle_each_iteration=True)
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
