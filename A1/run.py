#!/usr/bin/env python3
import argparse
import itertools
import os.path
import numpy as np
from functools import partial

from tqdm import tqdm
tqdm.pandas()

from inverted_index.utils import generate_gs_schemes, dict_to_str, product_dict
from inverted_index.model import SMARTVectorizer, SMARTSearch
from inverted_index.preprocessing import preprocess_data, whitespace_tokenize, remove_punctuation, UDPipeLemmatizer, \
    MyWordNetLemmatizer, remove_stopwords
from inverted_index.parsing import load_documents, load_queries, query_result_to_trec

parser = argparse.ArgumentParser()
parser.add_argument('-q', "--query", default='topics-train_en.xml', type=str, help="File including query topics.")
parser.add_argument('-d', "--documents", default='documents_en.lst', type=str, help="File including document filenames.")
parser.add_argument('-r', "--run-name", default='run-2_en', type=str, help="String identifying the experiment.")
parser.add_argument('-o', "--output", default='run-2_train_en.res', type=str, help="Output file.")
parser.add_argument('-s', "--weighting-scheme", default='dtb.dtb', type=str, help="Weighting scheme for inverted index."
                                                                                  " (default: nnc.nnc = \"baseline\").")
parser.add_argument('-g', "--grid-search", default=False, type=bool, help="Perform grid-search over params.")
parser.add_argument('-l', "--lang", default=None, type=str, help="Use language model for English/Czech.")
parser.add_argument('-D', "--use_description", default=True, type=bool, help="Use query description for retrieval.")


model_param_grid = {
    'slope': [0.1, 0.2, 0.5],
    'max_df': [0.2, 0.5, 0.8, 1.0],
    'min_df': [1, 5, 10, 25]
}

search_param_grid = {
    'n_pseudo_relevance': [5],
    'alpha': [0.5],
    'beta': [0.1],
    'gamma': [0.1]
}

data_transforms_cs = [UDPipeLemmatizer(), remove_punctuation, partial(remove_stopwords, lang='czech')]
query_transforms_cs = [UDPipeLemmatizer().udpipe_tokenize_lemmatize, remove_punctuation, partial(remove_stopwords, lang='czech')]
model_params_cs = {
    'weighting_scheme': 'dtb.dtb', 'slope': 0.1, 'min_df': 5, 'max_df': 0.5,
}
search_params_cs = {
    'n_pseudo_relevance': 5, 'alpha': 0.5, 'beta': 0.1, 'gamma': 0.1
}

data_transforms_en = [MyWordNetLemmatizer(), partial(remove_stopwords, lang='english')]
query_transforms_en = [MyWordNetLemmatizer().lemmatize_text, partial(remove_stopwords, lang='english')]
model_params_en = {
    'weighting_scheme': 'dtb.dtb', 'slope': 0.1, 'min_df': 5, 'max_df': 0.8,
}
search_params_en = {
    'n_pseudo_relevance': 5, 'alpha': 1, 'beta': 0.1, 'gamma': 0.1
}


def main(args):
    if args.lang is not None:
        lang = args.lang
    else:
        print('No language was specified, language will be inferred from the documents dir.')
        lang = 'cs' if 'cs' in args.documents else 'en'
    print(f'Using models and transforms for language: {lang}')
    data_transforms = data_transforms_cs if lang == 'cs' else data_transforms_en
    query_transforms = query_transforms_cs if lang == 'cs' else query_transforms_en
    print('#' * 22)
    print(f'#{"DATA PREPROCESSING":^20}#')
    print('#' * 22)
    print('Loading documents...')
    document_df = load_documents(args.documents)
    print('Pre-processing data...')
    document_df = preprocess_data(document_df, data_transforms)

    print('Loading queries')
    query_df = load_queries(args.query)
    print('Pre-processing data...')
    if not args.use_description:
        query_df['query'] = query_df['title'].apply(query_transforms[0])
    else:
        query_df['query'] = query_df['title'] + '\n' + query_df['desc']
        for transform in query_transforms:
            query_df['query'] = query_df['query'].apply(transform)
    print()

    print('#' * 22)
    print(f'#{"MODEL TRAINING":^20}#')
    print('#' * 22)
    if args.grid_search:
        for model_params in product_dict(**model_param_grid):
            output_fn = args.output.split('.')[0] + '_' + dict_to_str(model_params) + '.res'
            print(f'Fitting vector model for params: {model_params}')
            model = SMARTVectorizer(**model_params)
            model.fit(document_df['DATA'])

            weighting_schemes = [args.weighting_scheme] if args.weighting_scheme is not None else generate_gs_schemes(match_query=True)
            for i, ws in enumerate(weighting_schemes):
                print(f'Using weighting scheme: {ws} ({i + 1}/{len(weighting_schemes)})')
                gs_output_fn = output_fn if len(weighting_schemes) == 1 \
                    else output_fn.rsplit('.', maxsplit=1)[0] + '_' + ws.replace('.', '_') + '.res'
                print('Transforming documents and queries')
                document_weights = model.transform(document_df['DATA'], weighting_scheme=ws)
                query_weights = model.transform(query_df['query'], query=True)

                for search_params in product_dict(**search_param_grid):
                    print(f'Retrieving most similar documents using: {search_params}')
                    s_gs_output_fn = gs_output_fn.rsplit('.', maxsplit=1)[0] + '_' + dict_to_str(search_params) + '.res'
                    if os.path.exists(s_gs_output_fn):
                        continue
                    search = SMARTSearch(**search_params)

                    query_results = process_queries(search, document_weights, query_weights, document_df, query_df, args.run_name)
                    store_results(query_results, s_gs_output_fn)
    else:
        model_params = model_params_cs if lang == 'cs' else model_params_en
        search_params = search_params_cs if lang == 'cs' else search_params_en

        model = SMARTVectorizer(**model_params)
        model.fit(document_df['DATA'])

        document_weights = model.transform(document_df['DATA'])
        query_weights = model.transform(query_df['query'], query=True)

        search = SMARTSearch(**search_params)
        query_results = process_queries(search, document_weights, query_weights, document_df, query_df, args.run_name)
        store_results(query_results, args.output)


def process_queries(search, document_weights, query_weights, document_df, query_df, run_name):
    query_results = []
    for i, query_weight in enumerate(query_weights):
        most_relevant = search.get_k_most_relevant(query_weight, document_weights)
        query_results.append(query_result_to_trec(query_df.iloc[i], most_relevant, document_df, run_name))
    return query_results


def store_results(query_results, output_fn):
    with open(output_fn, 'w') as of:
        for result in query_results:
            of.write(result)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
