import shap
import math
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import lasso_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from autotune.utils.fanova import fANOVA
from autotune.utils.config_space import ConfigurationSpace
from autotune.utils.config_space.util import convert_configurations_to_array, config2df
from autotune.utils.logging_utils import get_logger
from autotune.knobs import initialize_knobs
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter, Constant
import pdb
import sys
import json
from collections import defaultdict


class KnobSelector(ABC):
    def __init__(self, selector_type):
        self.logger = get_logger(self.__class__.__name__)
        self.selector_type = selector_type

    def knob_selection(self,  config_space, history_container, num_hps, **kwargs):
        if self.selector_type == 'shap':
            return self.knob_selection_shap( config_space, history_container, num_hps, **kwargs)
        elif self.selector_type == 'fanova':
            return self.knob_selection_fanova(config_space, history_container, num_hps, **kwargs)
        elif self.selector_type == 'gini':
            return self.knob_selection_gini(config_space, history_container, num_hps, **kwargs)
        elif self.selector_type == 'ablation':
           return self.knob_selection_ablation(config_space, history_container, num_hps, **kwargs)
        elif self.selector_type == 'lasso':
            return self.knob_selection_lasso(config_space, history_container, num_hps, **kwargs)


    def knob_selection_shap(self, config_space, history_container, num_hps, use_imcub = True, prediction=True):
        columns = history_container.config_space_all.get_hyperparameter_names()

        X_df = config2df(history_container.configurations_all)
        X_df = X_df[columns]
        if not use_imcub:
            X_bench_df = config2df([history_container.config_space_all.get_default_configuration(), ])
        else:
            X_bench_df = config2df([history_container.incumbents[0][0]])

        X_bench_df = X_bench_df[columns]
        le = LabelEncoder()
        for col in list(X_df.columns):
            if isinstance(history_container.config_space_all.get_hyperparameters_dict()[col], CategoricalHyperparameter):
                le.fit(X_df[col])
                X_df[col] = le.transform(X_df[col])
                X_bench_df[col] = le.transform(X_bench_df[col])
            else:
                X_df[col] = X_df[col].astype('float')


        Y = -  np.array(history_container.get_transformed_perfs()).astype('float')
        Y_scaled = (Y - Y.min()) / (Y.max() - Y.min())

        if prediction:
            X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_scaled, test_size=0.05, random_state=0)
            model = LGBMRegressor()
            model.fit(X_train, Y_train)
            output_scaled = model.predict(X_test)
            output = output_scaled * (Y.max() - Y.min()) + Y.min()
            Y_test =  Y_test * (Y.max() - Y.min()) + Y.min()
            error = np.sqrt(mean_squared_error(Y_test, output ))
            r2_test = r2_score(Y_test, output)
            self.logger.info('The rmse of prediction of test set is: {:.2f}, {:.2%} of average tps, R2: {:.2%}'.format(
                error, error / Y_test.mean(), r2_test))

        model = LGBMRegressor()
        model.fit(X_df, Y_scaled)

        df = config2df(history_container.configurations_all)
        df = df[columns]
        df['objs'] = np.array(history_container.get_transformed_perfs())

        idx = df[df['objs'] <= df['objs'].quantile(0.1)].index.tolist()
        explainerModel = shap.TreeExplainer(model)
        shap_values = explainerModel.shap_values(np.array(X_df.loc[idx, :]))
        shap_value_default = explainerModel.shap_values(np.array(X_bench_df))[-1]
        delta = shap_values - shap_value_default
        delta = np.where(delta > 0, delta, 0)
        knobs_shap = np.average(delta, axis=0) * (Y.max() - Y.min())
        importance = {}
        for i in range(len(knobs_shap)):
            knob = columns[i]
            importance[knob] = knobs_shap[i]

        a = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        rank = [a[i][0] for i in range(num_hps)]

        hps = config_space.get_hyperparameters_dict()
        cs_new = ConfigurationSpace()

        for i in range(num_hps):
            self.logger.info("Top{}: {}, its shap value is {:.4}".format(i + 1, a[i][0], a[i][1]))
            knob = a[i][0]
            cs_new.add_hyperparameter(hps[knob])

        return cs_new, rank


    def knob_selection_fanova(self, config_space, history_container, num_hps):

        columns = history_container.config_space_all.get_hyperparameter_names()
        X = pd.DataFrame(history_container.configurations_all)
        X = X[columns]

        le = LabelEncoder()
        for col in list(X.columns):
            if isinstance(history_container.config_space_all[col], CategoricalHyperparameter):
                le.fit(X[col])
                X[col] = le.transform(X[col])
            else:
                X[col] = X[col].astype('float')

        Y = np.array(history_container.get_transformed_perfs()).astype('float')

        f = fANOVA(X=X, Y=Y, config_space=config_space)

        importance = {}
        for i in list(X.columns):
            value = f.quantify_importance((i,))[(i,)]['individual importance']
            if not math.isnan(value):
                importance[i] = value

        a = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        rank = [a[i][0] for i in range(num_hps)]

        hps = config_space.get_hyperparameters_dict()
        cs_new = ConfigurationSpace()

        for i in range(num_hps):
            if a[i][1] != 0:
                self.logger.info("Top{}: {}, its importance accounts for {:.4%}".format(i + 1, a[i][0], a[i][1]))
                knob = a[i][0]
                cs_new.add_hyperparameter(hps[knob])

        return cs_new, rank


    def knob_selection_gini(self, config_space, history_container, num_hps, prediction=True, logging=True):

        columns = history_container.config_space_all.get_hyperparameter_names()
        X = convert_configurations_to_array(history_container.configurations_all)
        Y = np.array(history_container.get_transformed_perfs())

        if prediction:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
            model = RandomForestRegressor()
            model.fit(X_train, Y_train)
            output = model.predict(X_test)
            error = np.sqrt(mean_squared_error(Y_test, output))
            r2_test = r2_score(Y_test, output)
            self.logger.info('The rmse of prediction of test set is: {:.2f}, {:.2%} of average tps, R2: {:.2%}'.format(
                error, error / Y_test.mean(), r2_test))

        model = RandomForestRegressor()
        model.fit(X, Y)
        feature_importance = model.feature_importances_

        importance = {}
        for i in range(len(feature_importance)):
            knob = columns[i]
            importance[knob] = feature_importance[i]

        a = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        rank = [a[i][0] for i in range(num_hps)]

        hps = config_space.get_hyperparameters_dict()
        cs_new = ConfigurationSpace()

        for i in range(num_hps):
            if logging:
                if a[i][1] > 0.01:
                    self.logger.info("Top{}: {}, its feature importance is {:.4%}".format(i + 1, a[i][0], a[i][1]))
            knob = a[i][0]
            cs_new.add_hyperparameter(hps[knob])

        return cs_new, rank



    def knob_selection_ablation(self, config_space, history_container, num_hps):

        columns = history_container.config_space_all.get_hyperparameter_names()

        X_df = pd.DataFrame(history_container.configurations_all)
        X_df = X_df[columns]

        X_default_df = pd.DataFrame([history_container.config_space_all.get_default_configuration(), ])
        X_default_df = X_default_df[columns]

        le = LabelEncoder()
        for col in list(X_df.columns):
            if isinstance(history_container.config_space_all[col], CategoricalHyperparameter):
                le.fit(X_df[col])
                X_df[col] = le.transform(X_df[col])
                X_default_df[col] = le.transform(X_default_df[col])
            else:
                X_df[col] = X_df[col].astype('float')

        Y = np.array(history_container.get_transformed_perfs()).astype('float')
        model = RandomForestRegressor()
        model.fit(X_df.to_numpy(), Y)

        df = pd.DataFrame(history_container.configurations_all)
        df = df[columns]
        df['objs'] = np.array(history_container.get_transformed_perfs())
        idx = df[df['objs'] < df['objs'].quantile(0.1)].index.tolist()

        top_knobs = {k: 0 for k in columns}
        for id in idx:
            path = self.generate_path(
                np.array(X_df.loc[id, :]).reshape(1, -1),
                X_default_df.to_numpy().reshape(1, -1),
                model,
                int(num_hps/2)
            )
            knob_path = [columns[i] for i in path]
            for k in knob_path:
                top_knobs[k] = top_knobs[k] + 1

        a = sorted(top_knobs.items(), key=lambda x: x[1], reverse=True)
        rank = [a[i][0] for i in range(num_hps)]

        hps = config_space.get_hyperparameters_dict()
        cs_new = ConfigurationSpace()

        for i in range(num_hps):
            self.logger.info("Top{}: {}, it appears in ablation path {} times".format(i + 1, a[i][0], a[i][1]))
            knob = a[i][0]
            cs_new.add_hyperparameter(hps[knob])

        return cs_new, rank

    def generate_path(self, target, default, emp, top_num=10):
        # y_target = emp.predict(target)[0]
        # y_default = emp.predict(default)[0]
        # self.logger.debug("y_target: {}, y_default: {}".format(y_target, y_default))
        x_path = default.copy()
        path = []
        path_i = 0
        # count top 20 in ablation path
        for j in range(0, top_num):
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
            # self.logger.debug("Top {}: {}, default value: {}, target value: {}, TPS: {}".format(j, knobsName[path_i],
            #                                                                         int(x_path[0][path_i]),
            #                                                                         int(target[0][path_i]), y_max))
            path.append(path_i)
            x_path[0][path_i] = target[0][path_i]

        return path



    def knob_selection_lasso(self, config_space, history_container, num_hps):

        columns = history_container.config_space_all.get_hyperparameter_names()
        X = convert_configurations_to_array(history_container.configurations_all)
        Y = np.array(history_container.get_transformed_perfs())
        alphas, coefs, _ = lasso_path(X, Y)

        feature_importance = []
        for i, feature_path in enumerate(coefs):
            entrance_step = 1
            for val_at_step in feature_path:
                if val_at_step == 0:
                    entrance_step += 1
                else:
                    break
            feature_importance.append(entrance_step)

        importance = {}
        for i in range(len(feature_importance)):
            knob = columns[i]
            importance[knob] = feature_importance[i]

        a = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        rank = [a[i][0] for i in range(num_hps)]

        hps = config_space.get_hyperparameters_dict()
        cs_new = ConfigurationSpace()

        for i in range(num_hps):
            self.logger.info("Top{}: {}, its entrance step on lasso path is {}".format(i + 1, a[i][0], a[i][1]))
            knob = a[i][0]
            cs_new.add_hyperparameter(hps[knob])

        return cs_new, rank



    def get_shape_value(self, config_space, history_container, use_imcub=True):
        columns = history_container.config_space_all.get_hyperparameter_names()

        X_df = config2df(history_container.configurations_all)
        X_df = X_df[columns]
        if not use_imcub:
            X_bench_df = config2df([history_container.config_space_all.get_default_configuration(), ])
        else:
            X_bench_df = config2df([history_container.incumbents[0][0]])

        X_bench_df = X_bench_df[columns]
        le = LabelEncoder()
        for col in list(X_df.columns):
            if isinstance(history_container.config_space_all.get_hyperparameters_dict()[col],
                          CategoricalHyperparameter):
                le.fit(X_df[col])
                X_df[col] = le.transform(X_df[col])
                X_bench_df[col] = le.transform(X_bench_df[col])
            else:
                X_df[col] = X_df[col].astype('float')

        Y = -  np.array(history_container.get_transformed_perfs()).astype('float')
        Y_scaled = (Y - Y.min()) / (Y.max() - Y.min())

        model = LGBMRegressor()
        model.fit(X_df, Y_scaled)

        df = config2df(history_container.configurations_all)
        df = df[columns]
        df['objs'] = np.array(history_container.get_transformed_perfs())

        idx = df[df['objs'] <= df['objs'].quantile(0.1)].index.tolist()
        explainerModel = shap.TreeExplainer(model)
        shap_values = explainerModel.shap_values(np.array(X_df.loc[idx, :]))
        shap_value_default = explainerModel.shap_values(np.array(X_bench_df))[-1]
        delta = shap_values - shap_value_default
        delta = np.where(delta > 0, delta, 0)
        knobs_shap = np.average(delta, axis=0) * (Y.max() - Y.min())
        importance = {}
        for i in range(len(knobs_shap)):
            knob = columns[i]
            importance[knob] = knobs_shap[i]

        a = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        rank =  [a[i][0] for i in range(len(a))]
        values = dict()
        for key, imp in a:
            values[key] = imp

        return rank, values




    def knob_selection_gini_shap(self, config_space, history_container, num_hps):
        shap_rank, shap_values = self.get_shape_value(config_space, history_container)
        total_knob_num = len(history_container.config_space_all.get_hyperparameter_names())
        _, gini_rank = self.knob_selection_gini(config_space, history_container, total_knob_num, logging=False)
        knobs_added = list()
        knob_left = list()
        for knob in gini_rank:
            if len(knobs_added) >= num_hps:
                break
            if shap_values[knob] > 0:
                knobs_added.append(knob)
            else:
                self.logger.info("{} is estimated only has negative effect".format(knob))
                knob_left.append(knob)

        shap_num = min(10, int(num_hps/2))
        for knob in shap_rank[:shap_num]:
            if not knob in gini_rank:
                if len(knobs_added) >= num_hps:
                    knobs_added.pop()
                knobs_added.append(knob)
                self.logger.info("Add shap top {}: {} with shap value {}".format(shap_num, knob, shap_values[knob]))

        if len(knobs_added) < num_hps:
            for knob in shap_rank:
                if knob not in knobs_added and shap_values[knob] > 0 and len(knobs_added) < num_hps:
                    knobs_added.append(knob)



        if len(knobs_added) < num_hps:
            for knob in knob_left:
                if  len(knobs_added) < num_hps:
                    knobs_added.append(knob)

        hps = config_space.get_hyperparameters_dict()
        cs_new = ConfigurationSpace()

        for i in range(num_hps):
            knob = knobs_added[i]
            self.logger.info("Top{}: {}, its importance accounts for {:.4}".format(i + 1, knob, shap_values[knob]))
            cs_new.add_hyperparameter(hps[knob])

        return cs_new, knobs_added


    def save_rank(self, rank, incumb , knob_config_file, output_file):
        konb_template = initialize_knobs(knob_config_file, -1)
        out_knob = dict()
        i = 0
        for knob in rank:
            konb_template[knob]['important_rank'] = i + 1
            i = i + 1
            out_knob[knob] = konb_template[knob]
        for key in konb_template.keys():
            if key not in out_knob.keys() and key  in incumb.keys():
                konb_template[key]['important_rank'] = -1
                if konb_template[key]['type'] == 'integer' and   konb_template[key]['max'] > sys.maxsize:
                    konb_template[key]['default'] = incumb[key] * 1000
                else:
                    konb_template[key]['default'] = incumb[key]
                out_knob[key] = konb_template[key]
            else:
                out_knob[key] = konb_template[key]

        with open(output_file, 'w') as fp:
            json.dump(out_knob, fp, indent=4)



