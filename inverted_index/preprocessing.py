import os
import pickle
import re
import time
from string import punctuation
from typing import Callable, List
from conllu import parse
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize

import pandas as pd
import requests

from tqdm import tqdm
tqdm.pandas()


def remap_word(word):
    if word == 'Eura':
        return 'euro'
    else:
        return word


class UDPipeLemmatizer:
    def __init__(self):
        self.lemmas = None
        if os.path.exists('udpipe_lemmas.pkl'):
            with open('udpipe_lemmas.pkl', 'rb') as f:
                self.lemmas = pickle.load(f)

    def udpipe_tokenize_lemmatize(self, text):
        MAX_LINE_LENGTH = 100_000
        lines = list(filter(None, text.split('\n')))
        lemmas = []
        for line in lines:
            if len(line) == 0:
                continue
            i = 0
            max_len = MAX_LINE_LENGTH
            if len(line) > MAX_LINE_LENGTH:
                max_len = len(line[:MAX_LINE_LENGTH].rsplit('.', maxsplit=1)[0]) + 1
            while len(l := line[max_len * i: max_len * (i + 1)]) > 0:
                while True:
                    try:
                        response = requests.get(
                            f'http://localhost:8001/process?tokenizer&tagger&parser&data={l}').json()
                        break
                    except:
                        time.sleep(1)
                response = parse(response['result'])
                for sentence in response:
                    lemmas.extend([remap_word(word['lemma']) for word in sentence])
                i += 1
        return lemmas

    def __call__(self, document):
        doc_id = document['DOCNO']
        if self.lemmas is not None:
            return self.lemmas[doc_id]
        text = document['DATA']
        return self.udpipe_tokenize_lemmatize(text)


class MyWordNetLemmatizer:
    LEMMAS_FN = 'wordnet_lemmas.pkl'

    def __init__(self, lowercase=True):
        self.lemmatizer = WordNetLemmatizer()
        self.lowercase = lowercase
        if os.path.exists(self.LEMMAS_FN):
            with open(self.LEMMAS_FN, 'rb') as f:
                self.lemmas = pickle.load(f)
            self.use_preloaded = True
        else:
            self.lemmas = {}
            self.use_preloaded = False

    def lemmatize_text(self, text):
        lines = list(filter(None, text.split('\n')))
        lemmas = []
        for line in lines:
            pos_tags = pos_tag(word_tokenize(line))
            for word, tag in pos_tags:
                if self.lowercase:
                    word = word.lower()
                if tag.startswith('N'):
                    t = 'n'
                elif tag.startswith('V'):
                    t = 'v'
                elif tag.startswith('J'):
                    t = 'a'
                elif tag.startswith('R'):
                    t = 'r'
                else:
                    continue
                lemma = self.lemmatizer.lemmatize(word, pos=t)
                lemmas.append(lemma)
        return lemmas

    def __call__(self, document):
        text = document['DATA']
        doc_id = document['DOCNO']
        if self.use_preloaded:
            return self.lemmas[doc_id]

        doc_lemmas = self.lemmatize_text(text)
        self.lemmas[doc_id] = doc_lemmas

        return doc_lemmas


def remove_punctuation(x):
    if isinstance(x, str):
        return re.sub(f'[{punctuation}]', ' ', x)
    elif isinstance(x, list):
        return [word.strip() for word in x if word.strip() not in punctuation]
    elif isinstance(x, pd.Series):
        return [word.strip() for word in x['DATA'] if word.strip() not in punctuation]
    else:
        raise NotImplementedError()


def whitespace_tokenize(x):
    if isinstance(x, str):
        return x.split()
    elif isinstance(x, pd.Series):
        return x['DATA'].split()
    else:
        raise NotImplementedError()


def preprocess_data(df: pd.DataFrame, funcs: List[Callable]):
    for i, func in enumerate(funcs):
        df['DATA'] = df.progress_apply(func, axis=1)
        if isinstance(func, MyWordNetLemmatizer):
            with open(func.LEMMAS_FN, 'wb') as f:
                pickle.dump(func.lemmas, f)
    return df
