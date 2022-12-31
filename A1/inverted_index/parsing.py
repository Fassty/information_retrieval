import os
import pickle
from typing import List

import pandas as pd
import pickle

from bs4 import BeautifulSoup
from tqdm import tqdm


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
    stored_doc_file = f'raw_documents_{document_dir.split("_")[-1]}.pkl'
    if not os.path.exists(stored_doc_file):
        with open(document_list_file) as f:
            document_filenames = [doc.strip() for doc in f]

        documents = []
        for document_fn in document_filenames:
            document_path = os.path.join(document_dir, document_fn)
            documents.extend(_load_xml_to_dicts(document_path))

        with open(stored_doc_file, 'wb') as f:
            pickle.dump(documents, f)
    else:
        with open(stored_doc_file, 'rb') as f:
            documents = pickle.load(f)

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
