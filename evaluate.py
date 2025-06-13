import itertools
import math
from collections.abc import Callable

import numpy as np


def evaluate_on[T](fd, class_lut: dict[str, T], classes: list[T] | np.ndarray, chunk_size: int,
                   compute: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> (float, np.array):
    if isinstance(classes, list):
        classes = np.array(classes)

    import csv

    reader = csv.reader(fd)
    y_true = np.array([], dtype=np.str_)
    pred = np.array([], dtype=np.str_)

    while rows := list(itertools.islice(reader, chunk_size)):
        queries = np.array(list(row[-2] for row in rows))
        y_true = np.concatenate((y_true, list(class_lut[row[-1]] for row in rows)))
        pred = np.concatenate((pred, compute(queries, classes)))

    from sklearn.metrics import confusion_matrix, f1_score
    return f1_score(y_true, pred, average='weighted', labels=classes), confusion_matrix(y_true, pred, labels=classes)


def posts(fd, chunk_size: int, compute: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    return evaluate_on(
        fd=fd,
        class_lut={'积极': 'positive', '消极': 'negative', '中性': 'neutral'},
        classes=['positive', 'negative', 'neutral'],
        chunk_size=chunk_size,
        compute=compute
    )
