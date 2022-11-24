#!/usr/bin/env python3
import argparse
import os.path

from tqdm import tqdm

from inverted_index.utils import generate_gs_schemes
from inverted_index.model import SMARTVectorizer
from inverted_index.preprocessing import preprocess_data, tokenize, remove_punctuation, udpipe_tokenize_lemmatize
from inverted_index.parsing import load_documents, load_queries, query_result_to_trec

parser = argparse.ArgumentParser()
parser.add_argument('-q', "--query", default='topics-train_cs.xml', type=str, help="File including query topics.")
parser.add_argument('-d', "--documents", default='documents_cs.lst', type=str, help="File including document filenames.")
parser.add_argument('-r', "--run-name", default='run-1_cs', type=str, help="String identifying the experiment.")
parser.add_argument('-o', "--output", default='udpipe/run-1_train_cs.res', type=str, help="Output file.")
parser.add_argument('-s', "--weighting-scheme", default='ann.ann', type=str, help="Weighting scheme for inverted index."
                                                                                  " (default: nnc.nnc = \"baseline\").")
parser.add_argument('-g', "--grid-search", default=True, type=bool, help="Perform grid-search over schemes.")


def main(args):
    print('Loading documents...')
    document_df = load_documents(args.documents)
    print('Pre-processing data...')
    document_df = preprocess_data(document_df, [udpipe_tokenize_lemmatize, remove_punctuation])

    print('Loading queries')
    query_df = load_queries(args.query)
    query_df['title'] = query_df['title'].apply(tokenize)

    print('Fitting vector model')
    model = SMARTVectorizer(slope=0.2)
    model.fit(document_df['DATA'])

    weighting_schemes = [args.weighting_scheme] if not args.grid_search else generate_gs_schemes(match_query=True)
    for ws in weighting_schemes:
        print(f'Transforming using model: {ws.split(".")[0]}')
        document_weights = model.transform(document_df['DATA'], weighting_scheme=ws)

        print(f'Processing queries: {ws.split(".")[1]}')
        output_fn = args.output if len(ws) == 1 else args.output.split('.')[0] + '_' + ws.replace('.', '_') + '.res'
        if os.path.exists(output_fn):
            continue
        with open(output_fn, 'w') as of:
            for _, query in tqdm(query_df.iterrows(), colour='green', total=len(query_df)):
                most_relevant = model.get_k_most_relevant(query['title'], document_weights)
                of.write(query_result_to_trec(query, most_relevant, document_df, args.run_name))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
