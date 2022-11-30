from __future__ import annotations
from collections import Counter, defaultdict
from gensim.similarities.fastss import editdist
from typing import Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
# https://openclean.readthedocs.io/source/examples/knn.html

# Scalar values.
Scalar = Union[int, float, str]
Value = Union[Scalar, Tuple[Scalar]]

def normalized_edit_distance(val_1: str, val_2: str) -> float:
    edit_distance = editdist(val_1, val_2)
    return 1 - (float(edit_distance) / max(len(val_1), len(val_2)))

def similarity_is_satisfied(distance: float, threshold: float) -> bool:
    if distance > threshold:
        return True
    else:
        return False

def ngram(value: str, n: int) -> List[str]:
    result = list()
    for i in range(len(value) - (n - 1)):
        result.append(value[i: i + n])

    return result


class Cluster(Counter):
    def add(self, value: Value, count: Optional[int] = 1) -> Cluster:
        self[value] += count
        return self

    def suggestion(self) -> Value:
        return self.most_common(1)[0][0]


class kNNClusterer:
    def __init__(
        self,
        threshold: Optional[float] = 0.9,
        minsize: Optional[int] = 2,
        ngram_n: Optional[int] = 6
    ):
        self.threshold = threshold
        self.minsize = minsize
        self.ngram_n = ngram_n

    def clusters(self, values: Counter) -> List[Cluster]:
        if not values:
            return list()

        blocks = self._get_blocks(values)

        freq = values

        clusters = defaultdict(Cluster)
        for block in blocks:
            for i in range(len(block) -1):
                val_i = block[i]
                for j in range(i + 1, len(block)):
                    val_j = block[j]
                    if val_j in clusters.get(val_i, dict()):
                        continue
                    
                    if not similarity_is_satisfied(normalized_edit_distance(val_i, val_j), self.threshold):
                        continue
                    clusters[val_i].add(val_j, freq[val_j])
                    clusters[val_j].add(val_i, freq[val_i])
        # i > jで計算しているので、残りの部分を加算する
        for key in clusters.keys():
            clusters[key].add(key, freq[key])

        return self._get_clusters(clusters.values())

    def _get_blocks(self, values) -> Iterable[List]:
        blocks = defaultdict(list)
        for value in values:
            for key in set(ngram(value=value, n=self.ngram_n)):
                blocks[key].append(value)
        return blocks.values()

    def _get_clusters(self, clusters: Iterable[Cluster]) -> List[Cluster]:
        result = list()

        for cluster in clusters:
            if len(cluster) < self.minsize:
                continue
            result.append(cluster)
        return result


def knn_clusters(
    data: pd.Series,
    threshold: Optional[float] = 0.9,
    minsize: Optional[int] = 2,
    ngram_n: Optional[int] = 6
    ):
    # DataFrameの型を確認
    if not isinstance(data, pd.Series):
        raise ValueError('data should be pd.Series')

    # 列の型はstringのみ
    if not data.dtypes.name in ['string']:
        raise ValueError('data dtype should be string')

    if data.isnull().any():
        raise ValueError('data should not include null')

    if minsize < 2:
        raise ValueError('minsize should be greater than 1')


    values = Counter(data.tolist())

    return kNNClusterer(
        threshold=threshold,
        minsize=minsize,
        ngram_n=ngram_n
    ).clusters(values=values)
