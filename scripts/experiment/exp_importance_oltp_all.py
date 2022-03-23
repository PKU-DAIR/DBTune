import json
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pdb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import shap

path="/Users/changzhuo/projects/autotune/experiment/gen_knobs/moreworkload/"



if __name__ == '__main__':
    fL = os.listdir(path)
    knobL = []
    shapDir = {}
    knob_all = {}
    for f in fL:
        f_json = open(path+f)
        konb_template = json.load(f_json)
        print("{} {}".format(f, konb_template['general_log']['shap_value']))
        kL = []
        for i in range(5):
            kL.append(list(konb_template.keys())[i])
        knobL.append(kL)

        for k in konb_template.keys():
            if k in shapDir.keys():
                shapDir[k] = shapDir[k] + konb_template[k]['shap_value']
            else:
                shapDir[k] = konb_template[k]['shap_value']


    Us = set(knobL[0])
    Is = set(knobL[0])
    for s in knobL[1:]:
        Us = Us.union(set(s))
        Is = Is.intersection(set(s))


    m = sorted(shapDir.items(), key=lambda kv: (kv[1], kv[0]) ,reverse=True)
    for i in range(len(m)):
        k = m[i][0]
        tmp = konb_template[k]
        tmp['shap_value'] = m[i][1]
        tmp['important_rank'] = i
        knob_all[k] = tmp


    with open("gen_knobs/OLTP.json", 'w') as fp:
       json.dump(knob_all, fp, indent=4)