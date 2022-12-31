import pytest

import numpy as np

from inverted_index.model import SMARTVectorizer


@pytest.fixture
def example_docs():
    return [
        'This is a sentence about whales'.split(),
        'I am talking about kangaroos here'.split(),
        'And also beavers'.split(),
        'They are all animals'.split(),
        'Another sentence about beavers'.split(),
        'In this sentence sentence is twice'.split(),
        'beavers beavers I love beavers'.split()
    ]


@pytest.fixture
def example_vocab(example_docs):
    words = sorted(list(set().union(*example_docs)))
    return {w: i for i, w in enumerate(words)}


@pytest.fixture
def example_tf_matrix(example_docs, example_vocab):
    tf_matrix = np.zeros((len(example_docs), len(example_vocab)))
    for doc_id, doc in enumerate(example_docs):
        for word_id, word in enumerate(doc):
            tf_matrix[doc_id, example_vocab[word]] += 1
    return tf_matrix


@pytest.fixture
def example_df_matrix(example_docs, example_vocab):
    df_matrix = np.zeros(len(example_vocab))
    for doc in example_docs:
        for word_id, word in enumerate(example_vocab):
            if word in doc:
                df_matrix[word_id] += 1
    return df_matrix


@pytest.fixture
def example_idf_matrix(example_docs, example_df_matrix):
    return np.log(len(example_docs) / example_df_matrix)


@pytest.fixture
def example_smooth_idf_matrix(example_docs, example_df_matrix):
    return np.log(len(example_docs) / (example_df_matrix + 1)) + 1


@pytest.fixture
def example_prob_idf_matrix(example_docs, example_df_matrix):
    return np.log((len(example_docs) - example_df_matrix) / example_df_matrix)


@pytest.fixture
def example_augmented_tf_matrix(example_tf_matrix):
    aug_tf_matrix = np.zeros_like(example_tf_matrix)
    for i, row in enumerate(example_tf_matrix):
        aug_tf_matrix[i] = 0.5 + 0.5 * row / row.max()
    return aug_tf_matrix


@pytest.fixture
def example_log_tf_matrix(example_tf_matrix):
    m = example_tf_matrix
    return 1 + np.log(m, out=np.zeros_like(m), where=(m != 0))


@pytest.fixture
def example_afb_tf_matrix(example_tf_matrix):
    afb_tf_matrix = np.zeros_like(example_tf_matrix)
    for i, row in enumerate(example_tf_matrix):
        m = row.mean()
        afb_tf_matrix[i] = (1 + np.log(row, out=np.zeros_like(row), where=(row != 0))) / (1 + np.log(m))
    return afb_tf_matrix


@pytest.fixture
def example_double_log_tf_matrix(example_tf_matrix):
    m = example_tf_matrix
    return 1 + np.log(1 + np.log(m, out=np.zeros_like(m), where=(m != 0)))


def test_avg_u(example_docs):
    model = SMARTVectorizer()
    model.fit(example_docs)

    u_per_doc = list(map(lambda x: len(set(x)), example_docs))
    mean_u = sum(u_per_doc) / len(u_per_doc)

    b_per_doc = list(map(lambda x: sum(map(len, x)), example_docs))
    mean_b = sum(b_per_doc) / len(b_per_doc)

    assert pytest.approx(mean_u, abs=1e-10) == model._avg_u
    assert pytest.approx(mean_b, abs=1e-10) == model._avg_b


def test_tf(example_docs, example_tf_matrix):
    model = SMARTVectorizer()
    tf_matrix = model._calculate_tf(example_docs, fit=True).todense()

    assert np.array_equal(example_tf_matrix, tf_matrix)


def test_df(example_docs, example_df_matrix):
    model = SMARTVectorizer()
    tf_matrix = model._calculate_tf(example_docs, fit=True)
    df_matrix = model._calculate_df(tf_matrix)

    assert np.array_equal(example_df_matrix, df_matrix)


def test_idf(example_docs, example_tf_matrix, example_idf_matrix):
    model = SMARTVectorizer(weighting_scheme='nfn.nnn')
    model.fit(example_docs)
    idf_matrix = np.array(model._df_idf(example_docs).todense()).diagonal()

    assert np.array_equal(example_idf_matrix, idf_matrix)


def test_tfidf(example_docs, example_tf_matrix, example_idf_matrix):
    model = SMARTVectorizer(weighting_scheme='nfn.nnn')
    tfidf_matrix = model.fit_transform(example_docs).todense()

    example_tf_idf_matrix = np.zeros_like(example_tf_matrix)
    for i, row in enumerate(example_tf_matrix):
        example_tf_idf_matrix[i] = row * example_idf_matrix
    example_tf_idf_matrix_diag = example_tf_matrix @ np.diag(example_idf_matrix)

    assert np.array_equal(example_tf_idf_matrix, example_tf_idf_matrix_diag)
    assert np.array_equal(example_tf_idf_matrix, tfidf_matrix)


def test_smooth_tfidf(example_docs, example_tf_matrix, example_smooth_idf_matrix):
    model = SMARTVectorizer(weighting_scheme='ntn.nnn')
    smooth_tfidf_matrix = model.fit_transform(example_docs).todense()

    example_smooth_tfidf_matrix = example_tf_matrix @ np.diag(example_smooth_idf_matrix)
    assert np.array_equal(example_smooth_tfidf_matrix, smooth_tfidf_matrix)


def test_prob_tfidf(example_docs, example_tf_matrix, example_prob_idf_matrix):
    model = SMARTVectorizer(weighting_scheme='npn.nnn')
    prob_tfidf_matrix = model.fit_transform(example_docs).todense()

    example_prob_tfidf_matrix = example_tf_matrix @ np.diag(example_prob_idf_matrix)
    assert np.array_equal(example_prob_tfidf_matrix, prob_tfidf_matrix)


def test_aug_tf(example_docs, example_augmented_tf_matrix):
    model = SMARTVectorizer(weighting_scheme='afn.nnn')
    model.fit(example_docs)

    aug_tf_matrix = model._tf_augmented(example_docs).todense()

    assert np.array_equal(example_augmented_tf_matrix, aug_tf_matrix)


def test_logarithmic_tf(example_docs, example_log_tf_matrix):
    model = SMARTVectorizer(weighting_scheme='lfn.nnn')
    model.fit(example_docs)

    log_tf_matrix = model._tf_logarithmic(example_docs).todense()

    assert np.array_equal(example_log_tf_matrix, log_tf_matrix)


def test_afb_tf(example_docs, example_afb_tf_matrix):
    model = SMARTVectorizer(weighting_scheme='Lfn.nnn')
    model.fit(example_docs)

    afb_tf_matrix = model._tf_average_freq_based(example_docs).todense()

    assert np.array_equal(example_afb_tf_matrix, afb_tf_matrix)


def test_double_log_tf(example_docs, example_double_log_tf_matrix):
    model = SMARTVectorizer(weighting_scheme='dfn.nnn')
    model.fit(example_docs)

    dlog_tf_matrix = model._tf_double_logarithm(example_docs).todense()

    assert np.array_equal(example_double_log_tf_matrix, dlog_tf_matrix)
