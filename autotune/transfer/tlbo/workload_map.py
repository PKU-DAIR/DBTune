import os
import sys
import pdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from autotune.transfer.tlbo.base import BaseTLSurrogate
from autotune.utils.parser import get_action_data_json
from autotune.utils.binner import Bin
from autotune.knobs import knobDF2action
from autotune.gp import gp_predict
from autotune.utils.history_container import HistoryContainer
from openbox.utils.config_space import Configuration
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.config_space import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter


class WorkloadMapping(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False):
        super().__init__(config_space, source_hpo_data, seed,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'mapping'
        self.source_dict = {}
        self.scaler = StandardScaler()
        self.binner = Bin(bin_start=1)
        self.extract_source()

        self.scale = True
        self.num_sample = 50
        self.iteration_id = 0

    def extract_source(self):
        for i, container in enumerate(self.source_hpo_data):
            X_scaled = convert_configurations_to_array(container.configurations)
            IM = np.vstack(container.get_internal_metrics())
            self.source_dict[container.task_id] = {'X': X_scaled, 'IM': IM, 'pos': i }

        IM_all = np.vstack([item['IM'] for item in list(self.source_dict.values())])
        self.scaler.fit_transform(IM_all)
        self.binner.fit(IM_all)
        del IM_all

    def train(self, target_hpo_data: HistoryContainer):
        # get target X, y, im
        target_X_scaled = convert_configurations_to_array(target_hpo_data.configurations)
        target_y = target_hpo_data.get_transformed_perfs()
        target_IM = np.vstack(target_hpo_data.get_internal_metrics())

        target_IM = self.scaler.transform(target_IM)
        target_IM = self.binner.transform(target_IM)

        scores = {}
        for task_id, item in list(self.source_dict.items()):
            predictions = np.empty_like(target_IM)
            source_X_scaled = item['X']
            source_IM_scaled = self.scaler.transform(item['IM'])
            for j, col in enumerate(source_IM_scaled.T):
                col = col.reshape(-1, 1)
                predictions[:, j] = gp_predict(source_X_scaled, col, target_X_scaled, task_id, j)
            predictions = self.binner.transform(predictions)
            dists = np.sqrt(np.sum(np.square(np.subtract(predictions, target_IM)), axis=1))
            scores[task_id] = np.mean(dists)

        best_score = np.inf
        best_task_id = None
        for task_id, similarity_score in list(scores.items()):
            if similarity_score < best_score:
                best_score = similarity_score
                best_task_id = task_id
        self.logger.info('In iter-%d,' % self.iteration_id)
        self.logger.info('Matched TaskID: %s' % best_task_id)

        mapped_container = self.source_hpo_data[self.source_dict[best_task_id]['pos']]
        mapped_X_scaled = convert_configurations_to_array(mapped_container.configurations)
        mapped_y = mapped_container.get_transformed_perfs()
        new_X = np.vstack((target_X_scaled, mapped_X_scaled))
        new_y = np.concatenate((target_y, mapped_y))

        self.target_surrogate = self.build_single_surrogate(new_X, new_y, normalize='standardize')
        self.iteration_id += 1

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        return mu, var


