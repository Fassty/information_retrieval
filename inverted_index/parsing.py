import os
import pickle
from typing import List

import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm

from inverted_index.preprocessing import udpipe_tokenize_lemmatize


def _load_xml_to_dicts(document_path: str) -> List[dict]:
    with open(document_path) as f:
        xml_data = f.read()
    bs_data = BeautifulSoup(xml_data, 'xml')

    documents = []
    for doc in bs_data.find_all('DOC'):
        doc_no = doc.find('DOCNO').text
        doc_id = doc.find('DOCID').text
        document = dict(
            DOCNO=doc_no,
            DOCID=doc_id,
            DATA=doc.text
        )
        documents.append(document)
    return documents


def load_documents(document_list_file: str) -> pd.DataFrame:
    document_dir = os.path.splitext(document_list_file)[0]
    with open(document_list_file) as f:
        document_filenames = [doc.strip() for doc in f]

    documents = []
    for document_fn in document_filenames:
        document_path = os.path.join(document_dir, document_fn)
        documents.extend(_load_xml_to_dicts(document_path))

    if os.path.exists('udpipe_lemmas.pkl'):
        with open('udpipe_lemmas.pkl', 'rb') as f:
            ud_pipe_lemmas = pickle.load(f)
    else:
        ud_pipe_lemmas = {}
    for i, document in tqdm(enumerate(documents)):
        if document['DOCNO'] in ud_pipe_lemmas:
            continue
        lemmas = udpipe_tokenize_lemmatize(document['DATA'])
        ud_pipe_lemmas[document['DOCNO']] = lemmas
        if i % 50 == 0:
            with open('udpipe_lemmas.pkl', 'wb') as f:
                pickle.dump(ud_pipe_lemmas, f)
    with open('udpipe_lemmas.pkl', 'wb') as f:
        pickle.dump(ud_pipe_lemmas, f)
    return pd.DataFrame(documents)


def load_queries(query_topics_fn: str) -> pd.DataFrame:
    return pd.read_xml(query_topics_fn)


def query_result_to_trec(query, most_relevant, document_df, run_id):
    trec_output = []
    for rank, (doc_id, sim) in enumerate(zip(*most_relevant)):
        docno = document_df['DOCNO'].iloc[doc_id]
        trec_output.append(
            f'{query["num"]}\t{0}\t{docno}\t{rank}\t{sim}\t{run_id}'
        )
    return '\n'.join(trec_output)
