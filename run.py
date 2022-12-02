#!/usr/bin/env python3
import argparse
import os.path

from tqdm import tqdm
tqdm.pandas()

from inverted_index.utils import generate_gs_schemes
from inverted_index.model import SMARTVectorizer
from inverted_index.preprocessing import preprocess_data, whitespace_tokenize, remove_punctuation, UDPipeLemmatizer, MyWordNetLemmatizer
from inverted_index.parsing import load_documents, load_queries, query_result_to_trec

parser = argparse.ArgumentParser()
parser.add_argument('-q', "--query", default='topics-train_en.xml', type=str, help="File including query topics.")
parser.add_argument('-d', "--documents", default='documents_en.lst', type=str, help="File including document filenames.")
parser.add_argument('-r', "--run-name", default='run-1_en', type=str, help="String identifying the experiment.")
parser.add_argument('-o', "--output", default='wordnet_slope/run-1_wordnet.res', type=str, help="Output file.")
parser.add_argument('-s', "--weighting-scheme", default='dtb.dtb', type=str, help="Weighting scheme for inverted index."
                                                                                  " (default: nnc.nnc = \"baseline\").")
parser.add_argument('-g', "--grid-search", default=False, type=bool, help="Perform grid-search over schemes.")
parser.add_argument('-f', "--fine-tune", default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], type=list, help="Fine-tune the slope parameter.")


def main(args):
    print('Loading documents...')
    document_df = load_documents(args.documents)
    print('Pre-processing data...')
    document_df = preprocess_data(document_df, [MyWordNetLemmatizer()])

    print('Loading queries')
    query_df = load_queries(args.query)
    query_df['title'] = query_df['title'].apply(MyWordNetLemmatizer().lemmatize_text)

    print('Fitting vector model')
    model = SMARTVectorizer(slope=0.2)
    model.fit(document_df['DATA'])

    if args.grid_search or args.fine_tune is None:
        grid_search_schemes(args, document_df, model, query_df)
    else:
        for slope in tqdm(args.fine_tune):
            model.slope = slope
            output_fn = args.output.split('.')[0] + '_' + f's={slope}' + '.res'
            document_weights = model.transform(document_df['DATA'], weighting_scheme=args.weighting_scheme)
            store_results(document_df, document_weights, model, output_fn, query_df, args.run_name)


def grid_search_schemes(args, document_df, model, query_df):
    weighting_schemes = [args.weighting_scheme] if not args.grid_search else generate_gs_schemes(match_query=True)
    with tqdm(total=len(weighting_schemes), colour='green') as pbar:
        for ws in weighting_schemes:
            output_fn = args.output if len(weighting_schemes) == 1 \
                else args.output.split('.')[0] + '_' + ws.replace('.', '_') + '.res'
            if os.path.exists(output_fn):
                continue
            document_weights = model.transform(document_df['DATA'], weighting_scheme=ws)
            store_results(document_df, document_weights, model, output_fn, query_df, args.run_name)
            pbar.update(1)


def store_results(document_df, document_weights, model, output_fn, query_df, run_name):
    with open(output_fn, 'w') as of:
        for _, query in query_df.iterrows():
            most_relevant = model.get_k_most_relevant(query['title'], document_weights)
            of.write(query_result_to_trec(query, most_relevant, document_df, run_name))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
