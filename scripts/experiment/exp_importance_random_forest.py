"""
Usage:Given your current konbs, this script will give you knobs importance and tuning suggestion based on decision tree

Training example:
python  tree_knob_recommend.py    --knobs=knobs28_v1.json
--train_log='log/train_ddpg_1566263747.log  log/train_ddpg_1573191796.log log/train_ddpg_1573892822.log'
--gen_tree_graph=False"

Inferance example:
python  tree_knob_recommend.py    --knobs=knobs28_v1.json
--params=tree/15XXXX.plk
--gen_tree_graph=False"


You must specify knobs.
You must specify either train_log or params.
You can specify tree_max_depth and gen_tree_graph (optional).
"""
import matplotlib.pyplot as plt
import json
from sklearn import metrics
from pathlib import Path
import argparse
import seaborn as sns
#import pydotplus
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split
import re
import sys
import ast
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import pdb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
regression = True

y_variable = 'tps'
output_file = 'gen_knobs/SYSBENCH_randomforest.json'

def load_json_res(file, konb_template):
    f = open(file)
    lines = f.readlines()
    dicL = []
    for line in lines:
        if '{' in line:
            line = line[line.index('{'):]
        tmp = line.split('|')
        json_str = tmp[0]
        knob = eval(json_str)
        knob_leave = {}
        for k in konb_template.keys():
            knob_leave[k] = str(knob[k])
        for i in range(1, len(tmp)-2):
            key = tmp[i].split('_')[0]
            value = float(tmp[i].split('_')[1])
            knob_leave[key] = value
        dicL.append(knob_leave)
    df = pd.DataFrame.from_dict(dicL)

    return df

def plot_dis(df):
    plt.figure()
    sns.set_style("ticks")
    '''sns.distplot(df, hist=False, kde=True, rug=True,  # 选择是否显示条形图、密度曲线、观测的小细条（边际毛毯）
                 kde_kws={"color": "lightcoral", "lw": 1.5, 'linestyle': '--'},  # 设置选择True的条件(其密度曲线颜色、线宽、线形)
                 rug_kws={'color': 'lightcoral', 'alpha': 1, 'lw': 2, }, label='TPS')'''
    sns.boxplot(y=df, label='TPS')
    plt.savefig('plot/tps.png')

if __name__ == '__main__':
    f_json = sys.argv[1]
    fn = sys.argv[2]
    f_json = open(f_json)
    konb_template = json.load(f_json)
    action_df = load_json_res(fn, konb_template)
    #action_df = action_df.drop('large_pages',axis=1)
    if regression:

        action_df = action_df.dropna(axis=0, how='any', subset=['tps'])
        action_df = action_df[~action_df['tps'].isin([-1, -2, -3, -4])]  # 去掉sysbench不成功的样本
        action_df = action_df[action_df['tps'] > 0]
        #action_df = action_df.iloc[:500,:]
        X = action_df.iloc[:, :-14]
        le = preprocessing.LabelEncoder()
        for c in list(X.columns):
            if konb_template[c]['type'] == 'enum':
                le.fit(X[c])
                X[c] = le.transform(X[c])
            else:
                X[c] = X[c].astype('float')
        Y = action_df[y_variable].astype('float')
        knobsName = list(action_df.columns)
        knobsName.remove('cpu')
        knobsName.remove(y_variable)
        ###############
        # for k in knobsName:
        #     plt.figure()
        #     plt.scatter(action_df[k],action_df['cpu'])
        #     plt.xlabel(k)
        #     plt.ylabel('cpu')
        #     plt.savefig('plot_cpu/{}_all.png'.format(k))
        #     plt.close()

        ################
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
        Y_test = np.array(Y_test)
        X_test = np.array(X_test)

        model = RandomForestRegressor()
        model.fit(X_train, Y_train)
        importance_raw = model.feature_importances_

        output = model.predict(X_test)
        error = np.sqrt(mean_squared_error(Y_test, output))
        r2 = r2_score(Y_test, output)
        print('The rmse of prediction of test set is: {:.2f}, {:.2%} of average tps, R2: {:.2%} '.format(error, error / Y_test.mean(), r2))
        output_train = model.predict(X_train)
        error_train = np.sqrt(mean_squared_error(Y_train, output_train))
        r2_train = r2_score(Y_train, output_train)
        print('The rmse of prediction of train set is: {:.2f}, {:.2%} of average tps, R2: {:.2%}'.format(error_train, error_train / Y_train.mean(), r2_train))
        dynamicL, staticL = [], []
        for key in konb_template.keys():
            if konb_template[key]['dynamic'] == 'Yes':
                dynamicL.append(key)
            else:
                staticL.append(key)
        importance = {}
        for i in range(0, importance_raw.shape[0]):
            importance[knobsName[i]] = importance_raw[i]
        a = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        #print('Knobs importance rank:')
        dy_sum, st_sum = 0, 0
        important_knobL = []
        for i in range(0, len(importance)):
            if a[i][1] != 0:
                #print("Top{}: {}, its importance accounts for {:.4%}\n".format(i + 1, a[i][0],
                  #                                                      a[i][1] / importance_raw.sum()))
                if a[i][0] in dynamicL:
                    dy_sum = dy_sum + a[i][1]
                else:
                    st_sum = st_sum + a[i][1]

                important_knobL.append(a[i][0])
                if i > 10:
                    continue
                '''
                plt.figure()
                # pdb.set_trace()
                plt.scatter(np.array(X[a[i][0]]), np.array(action_df['tps'].astype('float')))
                plt.xlabel(a[i][0])
                plt.ylabel('tps')
                plt.savefig('plot/{}_tps.png'.format(a[i][0]))
                plt.close()'''

        print ((dy_sum, st_sum))
        #print (important_knobL)
        i = 1
        out_knob = {}
        for key in important_knobL:
            konb_template[key]['important_rank'] = i
            out_knob[key] = konb_template[key]
            i = i + 1

        with open(output_file, 'w') as fp:
            json.dump(out_knob, fp, indent=4)


    else:
        action_df = action_df.fillna(0)
        action_df.tps = action_df.tps.mask(action_df.tps < 0, 0)# 去掉sysbench不成功的样本
        action_df.tps = action_df.tps.mask(action_df.tps > 0, 1)

        X = action_df.iloc[:, :-14]

        le = preprocessing.LabelEncoder()
        for c in list(X.columns):
            if konb_template[c]['type'] == 'enum':
                le.fit(X[c])
                X[c] = le.transform(X[c])
            else:
                X[c] = X[c].astype('float')
        Y = action_df['tps'].astype('float')
        knobsName = list(action_df.columns)
        knobsName.remove('cpu')
        knobsName.remove('tps')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
        Y_test = np.array(Y_test)
        X_test = np.array(X_test)
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)
        importance_raw = model.feature_importances_

        output = model.predict(X_test)
        accuracy = metrics.accuracy_score(Y_test, output)
        auc = metrics.roc_auc_score(Y_test, output)
        print('The accuracy of prediction of test set is: {:.2f}, AUC is {:.2f}'.format(accuracy, auc))
        output = model.predict(X_train)
        accuracy = metrics.accuracy_score(Y_train, output)
        auc = metrics.roc_auc_score(Y_train, output)
        print('The accuracy of prediction of train set is: {:.2f}, AUC is {:.2f}'.format(accuracy, auc))

        dynamicL, staticL = [], []
        for key in konb_template.keys():
            if konb_template[key]['dynamic'] == 'Yes':
                dynamicL.append(key)
            else:
                staticL.append(key)


        importance = {}
        for i in range(0, importance_raw.shape[0]):
            importance[knobsName[i]] = importance_raw[i]
        a = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print('Knobs importance rank:')
        pdb.set_trace()
        dy_sum, st_sum = 0, 0
        for i in range(0, len(importance)):
            if a[i][1] != 0:
                print("Top{}: {}, its importance accounts for {:.4%}\n".format(i + 1, a[i][0],
                                                                               a[i][1] / importance_raw.sum()))
                if a[i][0] in dynamicL:
                    dy_sum = dy_sum + a[i][1]
                else:
                    st_sum = st_sum + a[i][1]

                if i > 10:
                    continue
                plt.figure()
                # pdb.set_trace()
                plt.scatter(np.array(X[a[i][0]]), np.array(action_df['tps'].astype('float')))
                plt.xlabel(a[i][0])
                plt.ylabel('tps')
                plt.savefig('plot/{}_tps.png'.format(a[i][0]))
                plt.close()
        print ((dy_sum, st_sum))
        pdb.set_trace()
