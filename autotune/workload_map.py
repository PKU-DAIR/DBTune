import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .utils.binner import Bin
from .knobs import logger, knobDF2action
from .utils.parser import  get_action_data_json
from .gp import gp_predict
from autotune.knobs import knob2action


class WorkloadMapping:
    def __init__(self, mapping_model_dir, knobs_detail, y_variable):
        self.mapping_model_dir = mapping_model_dir
        self.knobs_detail = knobs_detail
        self.y_variable = y_variable


    def filter_knobs(self, knob_df):
        knobs_source = list(knob_df.columns)
        knob_df_filtered = pd.DataFrame(columns=list(self.knobs_detail.keys()))
        for knob in self.knobs_detail.keys():
            if knob not in knobs_source:
                return False, None
            knob_df_filtered[knob] = knob_df[knob]

        return True, knob_df_filtered

    def get_matched_data(self, target_X_scaled, y_target):
        matched_workload = self.map_workload(target_X_scaled, y_target)
        df1, df2, IM = get_action_data_json('{}/{}'.format(self.mapping_model_dir, matched_workload))
        _, knob_df = self.filter_knobs(df1)
        matched_action_df = knobDF2action(knob_df)
        matched_y = df2[self.y_variable]
        return matched_action_df, matched_y



    def map_workload(self, target_X_scaled, y_target):
        # load old workload data
        workload_list = os.listdir(self.mapping_model_dir)
        for f in workload_list:
            fn = os.path.join(self.mapping_model_dir, f)
            if os.path.isdir(fn):
                workload_list.remove(f)
                continue
        workload_list.sort()
        # for matching the history internal metrics which have 65 dimensiona
        # vesion1: workload mapping by predicting TPS
        '''  
        # normalize
        normalizer = StandardScaler()
        target_tps_scaled = normalizer.fit_transform(np.array(target_tps).reshape(-1, 1))
        target_tps_scaled = - target_tps_scaled
        scores = {}
        for workload_id, workload in enumerate(workload_list):
            # load old workload data
            old_X, old_tps = get_action_data_from_res('{}/{}'.format(self.output_log, workload))
            old_X_scaled = knobDF2action(old_X)
            old_record_num = len(old_tps)
            old_X_scaled = old_X_scaled[:old_record_num, :]

            normalizer = StandardScaler()
            old_tps_scaled = normalizer.fit_transform(np.array(old_tps).reshape(-1, 1))
            old_tps_scaled = - old_tps_scaled

            # predict tps at target_action_df
            tps_pred = get_pred_gp(old_X_scaled, old_tps_scaled, target_X_scaled)
            dists = np.sqrt(np.sum(np.square(
                np.subtract(tps_pred, target_tps_scaled)), axis=1))
            scores[workload] = np.mean(dists)
        '''
        # version2: workload mapping by predicting internal metrics
        # obtain all data for mapping
        workload_dir = {}
        for workload_name in workload_list:
            # load old workload data
            if os.path.getsize(os.path.join(self.mapping_model_dir, workload_name)) == 0:
                logger.info(('[Wokrload Mapping] {} is empty'.format(workload_name)))
                continue

            df1, _, IM = get_action_data_json(os.path.join(self.mapping_model_dir, workload_name), valid_IM=True)#df1, df2, np.vstack(internal_metricL)
            flag, knob_df = self.filter_knobs(df1)
            if not flag:
                continue
            workload_dir[workload_name] = {}
            workload_dir[workload_name]['X_matrix'] = knobDF2action(knob_df)
            workload_dir[workload_name]['y_matrix'] = IM

        # Stack all y matrices for preprocessing
        ys = np.vstack([entry['y_matrix'] for entry in list(workload_dir.values())])
        # Scale the  y values, then compute the deciles for each column in y
        y_scaler = StandardScaler()
        y_scaler.fit_transform(ys)
        y_binner = Bin(bin_start=1)
        y_binner.fit(ys)
        del ys

        # Now standardize the target's data and bin it by the deciles we just calculated
        y_target = y_scaler.transform(y_target)
        y_target = y_binner.transform(y_target)

        # workload mapping by predicting internal metrics
        scores = {}
        for workload_id, workload_entry in list(workload_dir.items()):
            predictions = np.empty_like(y_target)
            X_scaled = workload_entry['X_matrix']
            y_workload = workload_entry['y_matrix']
            y_scaled = y_scaler.transform(y_workload)
            for j, y_col in enumerate(y_scaled.T):
                # Using this workload's data, train a Gaussian process model
                # and then predict the performance of each metric for each of
                # the knob configurations attempted so far by the target.
                y_col = y_col.reshape(-1, 1)
                try:
                    predictions[:, j] = gp_predict(X_scaled, y_col, target_X_scaled, workload_id, j)
                except:
                    pdb.set_trace()
                # Bin each of the predicted metric columns by deciles and then
                # compute the score (i.e., distance) between the target workload
                # and each of the known workloads
            predictions = y_binner.transform(predictions)
            dists = np.sqrt(np.sum(np.square(
                np.subtract(predictions, y_target)), axis=1))
            scores[workload_id] = np.mean(dists)

        # Find the best (minimum) score
        best_score = np.inf
        best_workload = None
        for workload, similarity_score in list(scores.items()):
            if similarity_score < best_score:
                best_score = similarity_score
                best_workload = workload

        logger.info('[Wokrload Mapping] Score:{}'.format(str(scores)))
        logger.info('[Workload Mapping] Matched Workload: {}'.format(best_workload))

        return best_workload


    def get_matched_data_file(self, f):
        if not os.path.exists(f):
            return False, None, None

        f = open(f)
        lines = f.readlines()
        if len(lines) == 0:
            return False, None, None
        knobL, imL = [],[]
        for line in lines:
            knobL.append(eval(line.split('|')[0]))
            imL.append(eval(line.split('|')[1]))

        target_X_scaled = np.zeros((len(knobL), len(self.knobs_detail.keys())))
        y_target = np.zeros((len(knobL), 65))

        for i in range(len(knobL)):
            action = knob2action(knobL[i])
            im = imL[i]
            target_X_scaled[i, :] = action
            y_target[i, :] = im

        matched_workload = self.map_workload(target_X_scaled, y_target)

        df1, df2, IM = get_action_data_json('{}/{}'.format(self.mapping_model_dir, matched_workload))
        _, knob_df = self.filter_knobs(df1)
        matched_y = df2[self.y_variable]

        return True, knob_df, matched_y


    def get_runhistory_smac(self, f, cs):
        flag, matched_knob, matched_y = self.get_matched_data_file(f)
        if not flag:
            return False, None

        from smac.tae.execute_ta_run import StatusType
        from smac.runhistory.runhistory import RunHistory
        from smac.configspace import Configuration

        new_runhistory = RunHistory()
        all_data = {}
        all_data['config_origins'], all_data["configs"] = {}, {}
        all_data['data'] = []

        for i in range(1, matched_knob.shape[0] + 1):
            config = matched_knob.iloc[i-1, :]
            y = matched_y.iloc[i-1]
            all_data['config_origins'][str(i)] = "unknown"
            knobs = {}
            for k in self.knobs_detail.keys():
                if self.knobs_detail[k]['type'] == 'enum':
                    knobs[k] = str(config.loc[k])
                else:
                    knobs[k] = config.loc[k]
                if k == "innodb_online_alter_log_max_size":
                    knobs[k] = int(config.loc[k] / 10)

            all_data['configs'][str(i)] = knobs
            tmp = []
            tmp1 = [i, 'null', 0, 0.0]
            tmp2 = [-y, 260, 1, 0, 0, {}]
            tmp.append(tmp1)
            tmp.append(tmp2)
            all_data['data'].append(tmp)

        config_origins = all_data.get("config_origins", {})
        new_runhistory.ids_config = {
            int(id_): Configuration(
                cs, values=values, origin=config_origins.get(id_, None)
            ) for id_, values in all_data["configs"].items()
        }

        new_runhistory.config_ids = {config: id_ for id_, config in new_runhistory.ids_config.items()}

        new_runhistory._n_id = len(new_runhistory.config_ids)

        # important to use add method to use all data structure correctly
        for k, v in all_data["data"]:
            new_runhistory.add(config=new_runhistory.ids_config[int(k[0])],
                     cost=float(v[0]),
                     time=float(v[1]),
                     status=StatusType(v[2]),
                     instance_id=k[1],
                     seed=int(k[2]),
                     budget=float(k[3]) if len(k) == 4 else 0,
                     additional_info=v[3])

        return True, new_runhistory


    def get_XY_openbox(self, f, cs):
        flag, matched_knob, matched_y = self.get_matched_data_file(f)
        if not flag:
            return False, None

        from autotune.utils.config_space import Configuration
        from autotune.utils.config_space.util import convert_configurations_to_array
        configL, YL = [], []

        for i in range(0, matched_knob.shape[0] ):
            config = matched_knob.iloc[i, :]
            knobs = {}
            for k in self.knobs_detail.keys():
                if self.knobs_detail[k]['type'] == 'enum':
                    knobs[k] = str(config.loc[k])
                else:
                    knobs[k] = config.loc[k]
                if k == "innodb_online_alter_log_max_size":
                    knobs[k] = int(config.loc[k] / 10)
            config = Configuration(cs, values=knobs)
            configL.append(config)
            YL.append(matched_y.iloc[i])
        X = convert_configurations_to_array(configL)
        Y = np.array(YL)
        return X, Y












