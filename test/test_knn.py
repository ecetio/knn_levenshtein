from main.knn_levenshtein import knn_clusters

import pytest
import pandas as pd
from collections import Counter

def test_non():
    df = pd.Series(['abcdefgh', 'writers'],dtype='string')
    clusters = knn_clusters(data=df,threshold=0.9,minsize=2)
    assert(clusters == [])

def test_normal():
    df = pd.Series(['vintner', 'writers', 'winters'],dtype='string')
    clusters = knn_clusters(data=df,threshold=0.3,minsize=2,ngram_n=4)
    assert(len(clusters) == 2)
    assert(clusters[0] == Counter({'winters': 1, 'writers': 1}))
    assert(clusters[1] == Counter({'writers': 1, 'winters': 1}))

def test_invalid_df_error():
    df = pd.DataFrame({'a':['A','A','C','D'],
                       'b':['AA','AA','AA','AA'],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    with pytest.raises(ValueError) as e:
        knn_clusters(data=df,threshold=0.3,minsize=2,ngram_n=4)
    assert str(e.value) == "data should be pd.Series"

def test_invalid_dtype_error():
    df = pd.Series([123, 456, 789],dtype='float64')
    with pytest.raises(ValueError) as e:
        knn_clusters(data=df,threshold=0.3,minsize=2,ngram_n=4)
    assert str(e.value) == "data dtype should be string"

def test_null_error():
    df = pd.Series(['vintner', None, 'winters'],dtype='string')
    with pytest.raises(ValueError) as e:
        knn_clusters(data=df,threshold=0.3,minsize=2,ngram_n=4)
    assert str(e.value) == "data should not include null"

def test_minsize_error():
    df = pd.Series(['vintner', 'writers', 'winters'],dtype='string')
    with pytest.raises(ValueError) as e:
        knn_clusters(data=df,threshold=0.3,minsize=1,ngram_n=4)
    assert str(e.value) == "minsize should be greater than 1"


def test_suggestion():
    df = pd.Series(['vintner', 'writers', 'writers', 'winters'],dtype='string')
    clusters = knn_clusters(data=df,threshold=0.6,minsize=2,ngram_n=4)
    assert(len(clusters) == 2)
    assert(clusters[0] == Counter({'writers': 2, 'winters': 1}))
    assert(clusters[1] == Counter({'writers': 2, 'winters': 1}))
    assert(clusters[0].suggestion() == 'writers')
    assert(clusters[1].suggestion() == 'writers')


def test_not_satisfied():
    df = pd.Series(['writers', 'wintersa'],dtype='string')
    clusters = knn_clusters(data=df,threshold=0.7,minsize=2,ngram_n=4)
    assert(clusters == [])

def test_vali_valj_dup():
    df = pd.Series(['writers', 'winters', 'wintersa'],dtype='string')
    clusters = knn_clusters(data=df,threshold=0.8,minsize=2,ngram_n=4)
    assert(len(clusters) == 2)
    assert(clusters[0] == Counter({'wintersa': 1, 'winters': 1}))
    assert(clusters[1] == Counter({'winters': 1, 'wintersa': 1}))

def test_empty():
    df = pd.Series([],dtype='string')
    clusters = knn_clusters(data=df,threshold=0.8,minsize=2,ngram_n=4)
    assert(clusters == [])

def test_cluster_num_lower_than_minisize():
    df = pd.Series(['writers', 'winters', 'wintersa'],dtype='string')
    clusters = knn_clusters(data=df,threshold=0.8,minsize=3,ngram_n=4)
    assert(clusters == [])