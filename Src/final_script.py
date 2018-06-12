
from glob import glob
import pandas as pd
import os


def GetScore(csv_file):
    table = pd.read_csv(csv_file, index_col=0)
    return table['F1'].max()

if __name__ == '__main__':
    res = []
    files = glob("*/data_collector.csv")
    for f in files:
        score = GetScore(f)
        res += [(score, f.split('/')[0])]
    sorted_res = sorted(res, key=lambda x: x[0])
    best_model = sorted_res[0]
    os.mkdir("best")
    b = best_model[1]
    os.rename(b, os.path.join("best", b))