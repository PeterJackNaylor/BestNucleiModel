
from glob import glob
import pandas as pd
import os

tag = 'f1_score'

def GetScore(csv_file):
    table = pd.read_csv(csv_file, index_col=0)
    return table[tag].max()

if __name__ == '__main__':
    res = []
    files = glob("*/test_scores.csv")
    for f in files:
        score = GetScore(f)
        res += [(score, f.split('/')[0])]
    tab = pd.DataFrame(res)
    tab.set_index(1, inplace=True)
    tab.columns = [tag]
    sorted_res = sorted(res, key=lambda x: x[0])
    best_model = sorted_res[-1]
    os.mkdir("best")
    b = best_model[1]
    os.rename(b, os.path.join("best", b))
    tab.to_csv('recap.csv')
