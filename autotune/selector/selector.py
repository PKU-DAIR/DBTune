import abc
import shap
import numpy as np
from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.linear_model import lasso_path
from sklearn.ensemble import RandomForestRegressor
from openbox.utils.fanova import fANOVA
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space.util import convert_configurations_to_array
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter, Constant
import pdb

class KnobSelector(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def knob_selection_toy(self, config_space):
        hps = config_space.get_hyperparameters()
        cs_new = ConfigurationSpace()
        for i in range(int(len(hps)-2)):
            cs_new.add_hyperparameters([hps[i]])
        return cs_new, True

    def knob_selection(self, config_space, history_container, num_hps):
        pass


class SHAPSelector(KnobSelector):
    def knob_selection(self, config_space, history_container, num_hps):
        pdb.set_trace()
        X = convert_configurations_to_array(history_container.configurations_all)
        Y = np.array(history_container.get_transformed_perfs())

        # Fit a LightGBMRegressor with observations
        lgbr = LGBMRegressor()
        lgbr.fit(X, Y)
        explainer = shap.TreeExplainer(lgbr)
        shap_values = explainer.shap_values(X)
        feature_importance = np.mean(np.abs(shap_values), axis=0)

        hps = config_space.get_hyperparameters()
        keys = [hp.name for hp in hps]
        importance_list = []
        for i, hp_name in enumerate(keys):
            importance_list.append([hp_name, feature_importance[i]])
        importance_list.sort(key=lambda x: x[1], reverse=True)

        cs_new = ConfigurationSpace()
        for i in range(num_hps):
            hp_index = config_space.get_idx_by_hyperparameter_name(importance_list[i][0])
            cs_new.add_hyperparameter(hps[hp_index])
        return cs_new


class fANOVASelector(KnobSelector):
    def knob_selection(self, config_space, history_container, num_hps):
        X_from_dict = np.array([list(config.get_dictionary().values())
                                for config in history_container.configurations_all])
        X_from_array = np.array([config.get_array() for config in history_container.configurations])
        discrete_types = (CategoricalHyperparameter, OrdinalHyperparameter, Constant)
        discrete_idx = [isinstance(hp, discrete_types) for hp in config_space.get_hyperparameters()]
        X = X_from_dict.copy()
        X[:, discrete_idx] = X_from_array[:, discrete_idx]
        Y = np.array(history_container.get_transformed_perfs())
        f = fANOVA(X=X, Y=Y, config_space=config_space)

        # marginal for first parameter
        hps = config_space.get_hyperparameters()
        keys = [hp.name for hp in hps]
        importance_list = list()
        for key in keys:
            p_list = (key,)
            res = f.quantify_importance(p_list)
            individual_importance = res[(key,)]['individual importance']
            importance_list.append([key, individual_importance])
        importance_list.sort(key=lambda x: x[1], reverse=True)

        cs_new = ConfigurationSpace()
        for i in range(num_hps):
            hp_index = config_space.get_idx_by_hyperparameter_name(importance_list[i][0])
            cs_new.add_hyperparameter(hps[hp_index])
        return cs_new


class GiniSelector(KnobSelector):
    def knob_selection(self, config_space, history_container, num_hps):
        X = convert_configurations_to_array(history_container.configurations_all)
        Y = np.array(history_container.get_transformed_perfs())

        model = RandomForestRegressor()
        model.fit(X, Y)
        feature_importance = model.feature_importances_

        hps = config_space.get_hyperparameters()
        keys = [hp.name for hp in hps]
        importance_list = []
        for i, hp_name in enumerate(keys):
            importance_list.append([hp_name, feature_importance[i]])
        importance_list.sort(key=lambda x: x[1], reverse=True)

        cs_new = ConfigurationSpace()
        for i in range(num_hps):
            hp_index = config_space.get_idx_by_hyperparameter_name(importance_list[i][0])
            cs_new.add_hyperparameter(hps[hp_index])
        return cs_new


class AblationSelector(KnobSelector):
    def knob_selection(self, config_space, history_container, num_hps):
        X = convert_configurations_to_array(history_container.configurations_all)
        Y = np.array(history_container.get_transformed_perfs())

        default = []
        name = []
        for hp in config_space.get_hyperparameters():
            default.append(hp.normalized_default_value)
            name.append(hp.name)

        model = RandomForestRegressor()
        model.fit(X, Y)

        q = np.quantile(Y, 0.9)
        id_list = np.argwhere(Y > q)
        top_knobs = {}
        for id in id_list:
            path = self.generate_path(X[id], default, model, name)
            knob_path = [name[i] for i in path]
            for k in knob_path:
                if k not in top_knobs.keys():
                    top_knobs = 0
                top_knobs[k] = top_knobs + 1

        importance_list = sorted(top_knobs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        cs_new = ConfigurationSpace()
        for i in range(num_hps):
            hp_index = config_space.get_idx_by_hyperparameter_name(importance_list[i][0])
            cs_new.add_hyperparameter(hps[hp_index])
        return cs_new


    def generate_path(self, target, default, emp, knobsName):
        y_target = emp.predict(target)[0]
        y_default = emp.predict(default)[0]
        print("y_target: {}, y_default: {}".format(y_target, y_default))
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
            print("Top {}: {}, default value: {}, target value: {}, TPS: {}".format(j, knobsName[path_i],
                                                                                    int(x_path[0][path_i]),
                                                                                    int(target[0][path_i]), y_max))
            path.append(path_i)
            x_path[0][path_i] = target[0][path_i]

        return path


class LASSOSelector(KnobSelector):
    def knob_selection(self, config_space, history_container, num_hps):
        X = convert_configurations_to_array(history_container.configurations_all)
        Y = np.array(history_container.get_transformed_perfs())
        alphas, coefs, _ = lasso_path(X, Y)

        feature_rankings = [[] for _ in range(X.shape[1])]
        for target_coef_paths in coefs:
            for i, feature_path in enumerate(target_coef_paths):
                entrance_step = 1
                for val_at_step in feature_path:
                    if val_at_step == 0:
                        entrance_step += 1
                    else:
                        break
                feature_rankings[i].append(entrance_step)
        rankings = np.array([np.mean(ranks) for ranks in feature_rankings])
        rank_idxs = np.argsort(rankings)

        hps = config_space.get_hyperparameters()
        cs_new = ConfigurationSpace()
        for i in range(num_hps):
            hp_index = rank_idxs[i]
            cs_new.add_hyperparameter(hps[hp_index])
        return cs_new
