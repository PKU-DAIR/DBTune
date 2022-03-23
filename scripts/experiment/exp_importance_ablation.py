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
y_variable = 'tps' #'latency'
output_file = "gen_knobs/SYSBENCH_ablation.json"


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
            try:
                knob_leave[k] = str(knob[k])
            except:
                knob_leave[k] = str(konb_template[k]['default'])
        for i in range(1, len(tmp)-2):
            key = tmp[i].split('_')[0]
            value = float(tmp[i].split('_')[1])
            knob_leave[key] = value
        dicL.append(knob_leave)

    '''knob_default = {}
    for k in konb_template.keys():
        knob_default[k] = str(konb_template[k]['default'])
    knob_default['tps'] = 333
    dicL.append(knob_default)'''
    df = pd.DataFrame.from_dict(dicL)

    return df

def generate_path(target, default, emp, knobsName):
    y_target = emp.predict(target)[0]
    y_default = emp.predict(default)[0]
    print ("y_target: {}, y_default: {}".format(y_target, y_default))
    x_path = default.copy()
    path = []
    path_i = 0
    for j in range(0, 20):
        y_max = 0
        for i in range(0, default.shape[1]):
            if i in path:
                continue
            tmp = x_path.copy()
            tmp[0][i] = target[0][i]
            y_tmp = emp.predict(tmp)[0]
            if y_tmp > y_max:
                y_max = y_tmp
                path_i = i
        print ("Top {}: {}, default value: {}, target value: {}, TPS: {}".format(j, knobsName[path_i], int(x_path[0][path_i]), int(target[0][path_i]), y_max ))
        path.append(path_i)
        x_path[0][path_i] = target[0][path_i]

    return path



def generate_emp(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
    Y_test = np.array(Y_test)
    X_test = np.array(X_test)

    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    importance_raw = model.feature_importances_

    output = model.predict(X_test)
    error = np.sqrt(mean_squared_error(Y_test, output))
    r2_test = r2_score(Y_test, output)
    print('The rmse of prediction of test set is: {:.2f}, {:.2%} of average Y, R2: {:.2%}'.format(error, error / Y_test.mean(), r2_test))
    output_train = model.predict(X_train)
    error_train = np.sqrt(mean_squared_error(Y_train, output_train))
    r2_train = r2_score(Y_train, output_train)
    print('The rmse of prediction of train set is: {:.2f}, {:.2%} of average Y, R2: {:.2%}'.format(error_train, error_train / Y_train.mean(),r2_train))

    return model

if __name__ == '__main__':
    f_json = sys.argv[1]
    fn = sys.argv[2]
    f_json = open(f_json)
    konb_template = json.load(f_json)
    #konb_template = json.loads(konb_template)
    #action_df = get_action_data_json(fn, konb_template)
    action_df = load_json_res(fn, konb_template)

    #action_df = action_df[action_df['tps']!=action_df['tps'].max()]
    action_df = action_df.dropna(axis=0, how='any', subset=['tps'])
        # action_df.loc[action_df['tps'] != -1, 'tps'] = 0
    action_df = action_df[~action_df['tps'].isin([-1, -2, -3, -4])]  # 去掉sysbench不成功的样本
    action_df = action_df[action_df['tps'] > 0]
        #pdb.set_trace()
        #action_df = action_df.iloc[:500,:]
    defaultL = []
    X = action_df.iloc[:, :-14]
    le = preprocessing.LabelEncoder()
    for c in list(X.columns):
        default_v = konb_template[c]['default']
        if konb_template[c]['type'] == 'enum':
            try:
                le.fit(X[c])
                X[c] = le.transform(X[c])
                default_v = le.transform(np.array(str(default_v)).reshape(1,))
            except:
                pdb.set_trace()
        else:
            X[c] = X[c].astype('float')
        defaultL.append(int(default_v))

    #X = X.iloc[:-1, :]
    #action_df = action_df.iloc[:-1, :]
    Y = action_df[y_variable].astype('float')
    model = generate_emp(X, Y)

    knobsName = list(action_df.columns)
    knobsName.remove('cpu')
    knobsName.remove(y_variable)
    top_konbs = {}
    idL = action_df[action_df[y_variable]>action_df[y_variable].quantile(0.9)].index.tolist()
    if y_variable == 'lat':
        idL = action_df[action_df[y_variable] < action_df[y_variable].quantile(0.1)].index.tolist()
    print (idL)
    count = 0
    for id in idL:
        print("Path {}:".format(count))
        count = count + 1
        path = generate_path(np.array(X.loc[id, :]).reshape(1, -1), np.array(defaultL).reshape(1, -1), model, knobsName)
        knob_path = [knobsName[i] for i in path]
        print (knob_path)
        for k in knob_path:
            if k in top_konbs.keys():
                top_konbs[k] = top_konbs[k] + 1
            else:
                top_konbs[k] = 1
    m = sorted(top_konbs.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
    print (m)
    out_knob = {}
    for i in range(0, len(m)):
        knob = m[i][0]
        konb_template[knob]['important_rank'] = i+1
        out_knob[knob] =konb_template[knob]
    for key in konb_template.keys():
        if key not in out_knob.keys():
            konb_template[key]['important_rank'] = -1
            out_knob[key] = konb_template[key]

    with open(output_file, 'w') as fp:
        json.dump(out_knob, fp, indent=4)



