import re
import time
from string import punctuation
from typing import Callable, List
from conllu import parse

import pandas as pd
import requests


def udpipe_tokenize_lemmatize(text):
    MAX_LINE_LENGTH = 100_000
    lines = list(filter(None, text.split('\n')))
    lemmas = []
    for line in lines:
        i = 0
        max_len = MAX_LINE_LENGTH
        if len(line) > MAX_LINE_LENGTH:
            max_len = len(line[:MAX_LINE_LENGTH].rsplit('.', maxsplit=1)[0]) + 1
        while len(l := line[max_len * i: max_len * (i + 1)]) > 0:
            while True:
                try:
                    response = requests.get(f'http://localhost:8001/process?tokenizer&tagger&parser&data={l}').json()
                    break
                except:
                    print('AAAA')
                    time.sleep(1)
            response = parse(response['result'])
            for sentence in response:
                lemmas.extend([word['lemma'] for word in sentence])
            i += 1
    return lemmas


def remove_punctuation(x):
    if isinstance(x, str):
        return re.sub(f'[{punctuation}]', ' ', x)
    else:
        return [word.strip() for word in x if word.strip() not in punctuation]


def tokenize(x):
    return x.split()


def preprocess_data(df: pd.DataFrame, funcs: List[Callable]):
    #df['DATA'] = df[['TITLE', 'TEXT', 'HEADINGS', 'GEOGRAPHIES']].apply('\n'.join, axis=1)
    for i, func in enumerate(funcs):
        df['DATA'] = df['DATA'].apply(func)
        if i == 0:
            df.to_hdf('udpipe_save.h5', 'data')
    return df
