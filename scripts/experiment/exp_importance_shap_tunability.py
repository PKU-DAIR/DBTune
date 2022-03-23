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

y_variable = 'tps' #'lat'




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



def get_shap_values(action_df, konb_template):
    X = action_df.iloc[:, :-14]
    le = preprocessing.LabelEncoder()
    defaultL = []
    for c in list(X.columns):
        default_v = konb_template[c]['default']
        if konb_template[c]['type'] == 'enum':
            le.fit(X[c])
            X[c] = le.transform(X[c])
            default_v = le.transform(np.array(str(default_v)).reshape(1, ))
        else:
            X[c] = X[c].astype('float')
        defaultL.append(int(default_v))
    Y = action_df[y_variable].astype('float')
    Y = (Y - Y.mean())/Y.std()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    output = model.predict(X_test)
    error = np.sqrt(mean_squared_error(Y_test, output))
    r2_test = r2_score(Y_test, output)
    print('The rmse of prediction of test set is: {:.2f}, {:.2%} of average tps, R2: {:.2%}'.format(error,
                                                                                                    error / Y_test.mean(),
                                                                                                    r2_test))
    output_train = model.predict(X_train)
    error_train = np.sqrt(mean_squared_error(Y_train, output_train))
    r2_train = r2_score(Y_train, output_train)
    print('The rmse of prediction of train set is: {:.2f}, {:.2%} of average tps, R2: {:.2%}'.format(error_train,
                                                                                                     error_train / Y_train.mean(),
                                                                                                   r2_train))
    idL = action_df[action_df[y_variable] > action_df[y_variable].quantile(0.9)].index.tolist()
    if y_variable == 'lat':
        idL = action_df[action_df[y_variable] < action_df[y_variable].quantile(0.1)].index.tolist()
    shap_values = shap.TreeExplainer(model).shap_values(np.array(X.loc[idL, :]))

    return shap_values, X, Y, X_train, X_test, Y_train, Y_test, model, defaultL

if __name__ == '__main__':
    f_json = sys.argv[1]
    fn = sys.argv[2]
    f_json = open(f_json)
    konb_template = json.load(f_json)
    #konb_template = json.loads(konb_template)
    #action_df = get_action_data_json(fn, konb_template)
    action_df = load_json_res(fn, konb_template)
    action_df = action_df.dropna(axis=0, how='any', subset=['tps'])
    #action_df = action_df[~action_df['tps'].isin([-1, -2, -3, -4])]  # 去掉不成功的样本
    #action_df = action_df[action_df['tps'] > 0]
    action_df.loc[action_df['tps'] <= 0,'tps'] = action_df.loc[action_df['tps']>0,'tps'].min() * 0.9
    #action_df = action_df.iloc[:500,:]
    shap_values, X, Y, X_train, X_test, Y_train, Y_test, model, defaultL = get_shap_values(action_df, konb_template)
    explainerModel = shap.TreeExplainer(model)
    shap_values_Model = explainerModel.shap_values(np.array(defaultL).reshape(1,-1))
    shap_value_default = shap_values_Model[-1]
    shap_value_delta = shap_values - shap_value_default
    if y_variable == 'lat':
        shap_value_delta = - shap_value_delta
    knobs_shap = np.sum(shap_values - shap_value_default, axis=0)
    tunability_rank = np.argsort(-knobs_shap)
    out_knob = {}
    i = 0
    for id in tunability_rank :
        knob = X.columns[id]
        shap_value = knobs_shap[id]
        konb_template[knob]['important_rank'] = i + 1
        konb_template[knob]['shap_value'] = shap_value
        i = i + 1
        out_knob[knob] = konb_template[knob]
    for key in konb_template.keys():
        if key not in out_knob.keys():
            konb_template[key]['important_rank'] = -1
            out_knob[key] = konb_template[key]

    output_file = 'moreworkloads/'+ fn.split('/')[-1].split('.')[0] + '_shap.json'
    with open(output_file, 'w') as fp:
        json.dump(out_knob, fp, indent=4)

