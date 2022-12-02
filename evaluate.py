import os
import sys
import subprocess


if __name__ == '__main__':
    with open('run-1_wordnet_slope_eval.csv', 'w') as handle:
        handle.write('scheme,map\n')
        for f in os.listdir('wordnet_slope'):
            command = f'./eval/trec_eval -M1000 qrels-train_en.txt wordnet_slope/{f}'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            map_ = output.decode('ascii').split('\n')[5].split()[2]
            map_ = float(map_)
            handle.write(f'{"_".join(f.rsplit(".", maxsplit=1)[0].rsplit("_", maxsplit=2)[1:])},{map_:.5f}\n')
