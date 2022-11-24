import os
import sys
import subprocess


if __name__ == '__main__':
    with open('run-1_eval.csv', 'w') as handle:
        handle.write('scheme,map\n')
        for f in os.listdir('nolemma'):
            command = f'./eval/trec_eval -M1000 qrels-train_cs.txt nolemma/{f}'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            map_ = output.decode('ascii').split('\n')[5].split()[2]
            map_ = float(map_)
            handle.write(f'{"_".join(f.split(".")[0].rsplit("_", maxsplit=2)[1:])},{map_:.5f}\n')
