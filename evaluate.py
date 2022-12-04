import os
import sys
import subprocess


def append_to_result_file(model_file, result_file, lang):
    command = f'./eval/trec_eval -M1000 qrels-train_{lang}.txt {model_file}'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    results = output.decode('ascii').split('\n')
    results = list(filter(None, map(str.split, results)))
    results = {x[0]: x[2] for x in results}

    f = model_file.split('/')[-1]
    with open(result_file, 'a') as handle:
        handle.write(f'{f.rsplit(".", maxsplit=1)[0].split("_", maxsplit=1)[1]},{float(results["map"]):.5f},'
                     f'{float(results["P_10"]):.5f},{float(results["iprec_at_recall_0.00"]):.5f},'
                     f'{float(results["iprec_at_recall_0.10"]):.5f},{float(results["iprec_at_recall_0.20"]):.5f},'
                     f'{float(results["iprec_at_recall_0.30"]):.5f},{float(results["iprec_at_recall_0.40"]):.5f},'
                     f'{float(results["iprec_at_recall_0.50"]):.5f},{float(results["iprec_at_recall_0.60"]):.5f},'
                     f'{float(results["iprec_at_recall_0.70"]):.5f},{float(results["iprec_at_recall_0.80"]):.5f},'
                     f'{float(results["iprec_at_recall_0.90"]):.5f},{float(results["iprec_at_recall_1.00"]):.5f}\n')


if __name__ == '__main__':
    RESULT_FN = 'run-2_en_eval.csv'
    MODEL_DIR = None#'udpipe_queryexp'
    MODEL_NAME = 'run-2_train_en.res'
    LANG = 'en'
    with open(RESULT_FN, 'w') as handle:
        handle.write('scheme,map,P_10,PR_0,PR_10,PR_20,PR_30,PR_40,PR_50,PR_60,PR_70,PR_80,PR_90,PR_100\n')

    if MODEL_DIR is not None:
        for f in os.listdir(MODEL_DIR):
            model_file = os.path.join(MODEL_DIR, f)
            append_to_result_file(model_file, RESULT_FN, LANG)
    elif MODEL_NAME is not None:
        append_to_result_file(MODEL_NAME, RESULT_FN, LANG)
