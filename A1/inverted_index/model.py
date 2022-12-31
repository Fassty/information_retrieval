import numbers
from collections import defaultdict

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from functools import partial, lru_cache

from inverted_index.utils import logn, top_k_csr, divide_rows_csr
import time


class SMARTVectorizer:
    def __init__(
            self,
            *,
            weighting_scheme='nnc.nnc',
            log_base=np.e,
            slope=0.5,
            min_df=0,
            max_df=1.0
    ):
        self.weighting_scheme = weighting_scheme
        # Word to index mapping
        self.vocabulary = None
        # Average number of unique terms in document
        # Used for pivoted unique normalization
        self._avg_u = 0
        # Average document length
        # Used for pivoted character length normalization
        self._avg_b = 0
        self._log_base = log_base
        self._slope = slope
        self._min_df = min_df
        self._max_df = max_df
        self._tf_weighting_mapping = {
            # Binary weight
            'b': self._tf_binary_weight,
            # Raw term frequency
            'n': self._tf_natural,
            't': self._tf_natural,
            # Augmented normalized term frequency
            'a': self._tf_augmented,
            # Logarithmic term frequency
            'l': self._tf_logarithmic,
            # Average-term-frequency-based normalization
            'L': self._tf_average_freq_based,
            # Double logarithm
            'd': self._tf_double_logarithm
        }
        self._df_weighting_mapping = {
            # No document frequency weighting
            'x': self._df_ones,
            'n': self._df_ones,
            # Default IDF
            'f': self._df_idf,
            # Smooth IDF
            't': partial(self._df_idf, smooth=True),
            # Probabilistic IDF
            'p': partial(self._df_idf, probabilistic=True),
        }
        self._norm_mapping = {
            'n': self._identity,
            'c': self._l2_norm,
            'u': self._pivoted_unique,
            'b': self._pivoted_char_length
        }
        self.tf, self.df, self.idf = None, None, None

    @property
    def document_weighting(self):
        return self.weighting_scheme.split('.')[0]

    @property
    def query_weighting(self):
        return self.weighting_scheme.split('.')[1]

    def _calculate_tf(self, docs, fit=False):
        assert fit or self.vocabulary is not None, 'Model must be fitted first.'
        values = []
        indices = []
        indptr = [0]
        if fit:
            self.vocabulary = defaultdict()
            self.vocabulary.default_factory = self.vocabulary.__len__

        n = 1
        for doc in docs:
            counter = {}
            doc_len = 0
            n_terms = len(set(doc))
            for word in doc:
                # Ignore out-of-vocabulary words during inference
                if not fit and word not in self.vocabulary:
                    continue
                idx = self.vocabulary[word]
                if idx not in counter:
                    counter[idx] = 1
                else:
                    counter[idx] += 1
                doc_len += len(word)
            if fit:
                self._avg_b += (doc_len - self._avg_b) / n
                self._avg_u += (n_terms - self._avg_u) / n
            n += 1
            indices.extend(counter.keys())
            values.extend(counter.values())
            indptr.append(len(indices))
        values = np.array(values)
        indices = np.array(indices)
        indptr = np.array(indptr)

        tf = csr_matrix(
            (values, indices, indptr),
            shape=(len(docs), len(self.vocabulary)),
            dtype=np.int32
        )
        tf.sort_indices()

        if fit:
            self.vocabulary = dict(self.vocabulary)
            sorted_vocab = sorted(self.vocabulary.items())
            index_remap = np.empty(len(sorted_vocab), dtype=np.int32)
            for new_value, (word, old_value) in enumerate(sorted_vocab):
                self.vocabulary[word] = new_value
                index_remap[old_value] = new_value
            tf.indices = index_remap.take(tf.indices, mode='clip')

        return tf

    def _calculate_df(self, tf):
        if self.df is not None:
            return self.df
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.df = df
        return df

    def _remove_common(self, tf):
        n_docs = tf.shape[0]
        vocab_size = tf.shape[1]
        min_df = self._min_df if isinstance(self._min_df, numbers.Integral) else self._min_df * n_docs
        max_df = self._max_df if isinstance(self._max_df, numbers.Integral) else self._max_df * n_docs

        mask = np.ones(len(self.df), dtype=bool)
        mask &= self.df <= max_df
        mask &= self.df >= min_df

        new_indices = np.cumsum(mask) - 1
        for term, old_index in list(self.vocabulary.items()):
            if mask[old_index]:
                self.vocabulary[term] = new_indices[old_index]
            else:
                del self.vocabulary[term]
        tf = tf[:, np.where(mask)[0]]
        new_vocab_size = tf.shape[1]
        print(f'Vocab size reduction: {(1 - (new_vocab_size / vocab_size)) * 100:.2f}%')
        self.df = None
        self._calculate_df(tf)

    def _tf_binary_weight(self, docs):
        tf = self._calculate_tf(docs, fit=False)
        return tf.astype(bool).astype(int)

    def _tf_natural(self, docs):
        return self._calculate_tf(docs, fit=False)

    def _tf_augmented(self, docs):
        tf = self._calculate_tf(docs, fit=False)
        tf.data = 0.5 + 0.5 * divide_rows_csr(tf.data, tf.indices, tf.indptr, tf.max(axis=1).data)
        return tf

    def _tf_logarithmic(self, docs):
        tf = self._calculate_tf(docs, fit=False)
        tf.data = 1 + logn(tf.data, self._log_base)
        return tf

    def _tf_average_freq_based(self, docs):
        tf = self._calculate_tf(docs, fit=False)
        numer = 1 + logn(tf.data, self._log_base)
        denom = 1 + logn(tf.mean(axis=1).data, self._log_base)
        tf.data = numer
        tf.data = divide_rows_csr(tf.data, tf.indices, tf.indptr, denom)
        return tf

    def _tf_double_logarithm(self, docs):
        tf = self._calculate_tf(docs, fit=False)
        tf.data = 1 + logn(1 + logn(tf.data, self._log_base), self._log_base)
        return tf

    def _df_ones(self, docs):
        idf = diags(np.ones(len(self.vocabulary)), format='csr')
        return idf

    def _df_idf(self, docs, smooth=False, probabilistic=False):
        if self.idf is not None:
            return self.idf
        if smooth:
            idf = logn(len(docs) / (self.df + 1), self._log_base) + 1
        elif probabilistic:
            idf = logn((len(docs) - self.df) / self.df, self._log_base)
        else:
            idf = logn(len(docs) / self.df, self._log_base)
        idf = diags(idf)
        self.idf = idf
        return idf

    def _identity(self, docs, weights):
        return weights

    def _l2_norm(self, docs, weights):
        return normalize(weights, norm='l2', axis=1)

    def _calculate_u(self, docs):
        return np.array(list(map(lambda x: len(set(x)), docs)))

    def _calculate_b(self, docs):
        return np.array(list(map(lambda x: sum(map(len, x)), docs)))

    def _pivoted_unique(self, docs, weights):
        norm_factors = 1 - self._slope + self._slope * (self._calculate_u(docs) / self._avg_u)
        weights.data = divide_rows_csr(weights.data, weights.indices, weights.indptr, norm_factors)
        return weights

    def _pivoted_char_length(self, docs, weights):
        norm_factors = 1 - self._slope + self._slope * (self._calculate_b(docs) / self._avg_b)
        weights.data = divide_rows_csr(weights.data, weights.indices, weights.indptr, norm_factors)
        return weights

    def _calculate_tf_weights(self, docs, query=False):
        scheme = self.query_weighting if query else self.document_weighting
        tf_weighting_method = self._tf_weighting_mapping[scheme[0]]
        return tf_weighting_method(docs)

    def _calculate_df_weights(self, docs, query=False):
        scheme = self.query_weighting if query else self.document_weighting
        df_weighting_method = self._df_weighting_mapping[scheme[1]]
        return df_weighting_method(docs)

    def _normalize_weights(self, docs, weights, query=False):
        scheme = self.query_weighting if query else self.document_weighting
        normalization_method = self._norm_mapping[scheme[2]]
        return normalization_method(docs, weights)

    def fit(self, docs):
        tf = self._calculate_tf(docs, fit=True)
        self._calculate_df(tf)
        self._remove_common(tf)
        return self

    def transform(self, docs, weighting_scheme=None, query=False):
        if weighting_scheme:
            self.weighting_scheme = weighting_scheme
            self.idf = None
        tf_weights = self._calculate_tf_weights(docs, query=query)
        df_weights = self._calculate_df_weights(docs, query=query)
        document_weights = tf_weights * df_weights
        document_weights = self._normalize_weights(docs, document_weights, query=query)
        return document_weights

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)


class SMARTSearch:
    def __init__(
            self,
            *,
            n_pseudo_relevance: int = 0,
            alpha: float = 0,
            beta: float = 0,
            gamma: float = 0
    ):
        self._n_pseudo_relevance = n_pseudo_relevance
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def get_k_most_relevant(self, query_weights, document_weights, k=1000):
        if self._n_pseudo_relevance > 0:
            ranking = query_weights * document_weights.T
            top_k_indices, _ = top_k_csr(ranking.data, ranking.indices, ranking.indptr, self._n_pseudo_relevance)
            top_k_indices = top_k_indices.squeeze()

            mask = np.ones(document_weights.shape[0], dtype=bool)
            mask[top_k_indices] = False
            relevant = document_weights[top_k_indices].mean(axis=0)
            non_relevant = document_weights[mask].mean(axis=0)

            if self._alpha == self._beta == self._gamma == 0:
                # Use classic Rocchio $$q_{opt} = \mu(D_r) + (\mu(D_r) - \mu(D_{nr}))$$
                query = relevant + (relevant - non_relevant)
            else:
                # Rocchio SMART
                query = self._alpha * query_weights + self._beta * relevant - self._gamma * non_relevant
            query = csr_matrix(query)
        else:
            query = query_weights

        ranking = query * document_weights.T
        top_k_indices, top_k_sims = top_k_csr(ranking.data, ranking.indices, ranking.indptr, k)
        return top_k_indices.squeeze(), top_k_sims.squeeze()
