# License: MIT
import os
import pdb
import sys
import time
import json
import collections
from typing import List, Union
import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from autotune.utils.constants import MAXINT, SUCCESS
from autotune.utils.config_space import Configuration, ConfigurationSpace
from autotune.utils.logging_utils import get_logger
from autotune.utils.multi_objective import Hypervolume, get_pareto_front
from autotune.utils.config_space.space_utils import get_config_from_dict
from autotune.utils.visualization.plot_convergence import plot_convergence
from autotune.utils.transform import  get_transform_function
from openbox.utils.config_space.util import convert_configurations_to_array

Perf = collections.namedtuple(
    'perf', ['cost', 'time', 'status', 'additional_info'])

Observation = collections.namedtuple(
    'Observation', ['config', 'trial_state', 'constraints', 'objs', 'elapsed_time',  'iter_time','EM', 'IM', 'resource', 'info', 'context'])


def detect_valid_history_file(dir):
    if not os.path.exists(dir):
        return []
    files = os.listdir(dir)
    valid_files = []
    for f in files:
        try:
            with open(fn) as fp:
                all_data = json.load(fp)
        except Exception as e:
            continue
        data = all_data['data']
        valid_count = 0
        for item in data:
            if item['trial_state'] == 0:
                valid_count = valid_count + 1
            if valid_count > len(data)/2:
                valid_files.append(files)
                continue
    return valid_files


def load_history_from_filelist(task_id, fileL, config_space):

    data_mutipleL = list()
    for fn in fileL:
        try:
            with open(fn) as fp:
                all_data = json.load(fp)
        except Exception as e:
            print('Encountered exception %s while reading runhistory from %s. '
                  'Not adding any runs!', e, fn, )
            return

        info = all_data["info"]
        data = all_data["data"]
        data_mutipleL = data_mutipleL + data


    file_out = 'history_{}.json'.format(task_id)
    with open(file_out, "w") as fp:
        json.dump({"info": info, "data": data_mutipleL}, fp, indent=2)

    history_container = HistoryContainer(task_id, config_space=config_space)
    history_container.load_history_from_json(file_out)

    return history_container






class HistoryContainer(object):
    def __init__(self, task_id, num_constraints=0, config_space=None):
        self.task_id = task_id
        self.num_objs = 1
        self.num_constraints = num_constraints
        self.config_space = config_space  # for show_importance
        self.config_space_all = config_space
        self.logger = get_logger(self.__class__.__name__)

        self.info = None
        self.config_counter = 0
        self.data = collections.OrderedDict()  # only successful data
        self.data_all = collections.OrderedDict()
        self.incumbent_value = MAXINT
        self.incumbents = list()
        self.configurations = list()  # all configurations (include successful and failed)
        self.configurations_all = list()
        self.perfs = list()  # all perfs
        self.constraint_perfs = list()  # all constraints
        self.trial_states = list()  # all trial states
        self.elapsed_times = list()  # all elapsed times
        self.iter_times = list()
        self.external_metrics = list() # all external metrics
        self.internal_metrics = list() # all internal metrics
        self.resource = list() # all resource information
        self.contexts = list()

        self.update_times = list()  # record all update times

        self.successful_perfs = list()  # perfs of successful trials
        self.failed_index = list()
        self.transform_perf_index = list()

        self.global_start_time = time.time()
        self.scale_perc = 5
        self.perc = None
        self.min_y = None
        self.max_y = MAXINT

    def fill_default_value(self, config):
        values = {}
        for key in self.config_space_all._hyperparameters:
            if key in config.keys():
                values[key] = config[key]
            else:
                values[key] = self.config_space_all._hyperparameters[key].default_value

        c_new = Configuration(self.config_space_all, values)

        return c_new

    def update_observation(self, observation: Observation):
        self.update_times.append(time.time() - self.global_start_time)
        config = observation.config
        objs = observation.objs
        constraints = observation.constraints
        trial_state = observation.trial_state
        elapsed_time = observation.elapsed_time
        iter_time = observation.iter_time
        internal_metrics = observation.IM
        external_metrics = observation.EM
        resource = observation.resource
        info = observation.info
        context = observation.context

        if not self.info:
            self.info = info

        assert self.info == info

        self.configurations.append(config)
        self.configurations_all.append((self.fill_default_value(config)))
        if self.num_objs == 1:
            self.perfs.append(objs[0])
        else:
            self.perfs.append(objs)
        self.trial_states.append(trial_state)
        self.constraint_perfs.append(constraints)  # None if no constraint
        self.elapsed_times.append(elapsed_time)
        self.iter_times.append(iter_time)
        self.internal_metrics.append(internal_metrics)
        self.external_metrics.append(external_metrics)
        self.resource.append(resource)
        self.contexts.append(context)

        transform_perf = False
        failed = False
        if trial_state == SUCCESS and all(perf < MAXINT for perf in objs):
            if self.num_constraints > 0 and constraints is None:
                self.logger.error('Constraint is None in a SUCCESS trial!')
                failed = True
                transform_perf = True
            else:
                # If infeasible, transform perf to the largest found objective value
                feasible = True
                if self.num_constraints > 0 and any(c > 0 for c in constraints):
                    transform_perf = True
                    feasible = False

                if self.num_objs == 1:
                    self.successful_perfs.append(objs[0])
                else:
                    self.successful_perfs.append(objs)
                if feasible:
                    if self.num_objs == 1:
                        self.add(config, objs[0])
                    else:
                        self.add(config, objs)
                else:
                    self.add(config, MAXINT)

                self.perc = np.percentile(self.successful_perfs, self.scale_perc, axis=0)
                self.min_y = np.min(self.successful_perfs, axis=0).tolist()
                self.max_y = np.max(self.successful_perfs, axis=0).tolist()

        else:
            # failed trial
            failed = True
            transform_perf = True

        cur_idx = len(self.perfs) - 1
        if transform_perf:
            self.transform_perf_index.append(cur_idx)
        if failed:
            self.failed_index.append(cur_idx)

    def get_contexts(self):
        return np.vstack(self.contexts)


    def add(self, config: Configuration, perf):
        if config in self.data:
            self.logger.warning('Repeated configuration detected!')
            return

        self.data[config] = perf
        self.data_all[self.fill_default_value(config)] = perf

        if len(self.incumbents) > 0:
            if perf < self.incumbent_value:
                self.incumbents.clear()
            if perf <= self.incumbent_value:
                self.incumbents.append((config, perf))
                self.incumbent_value = perf
        else:
            self.incumbent_value = perf
            self.incumbents.append((config, perf))

    def get_transformed_perfs(self, transform=None):
        # set perf of failed trials to current max
        transformed_perfs = self.perfs.copy()
        for i in self.transform_perf_index:
            transformed_perfs[i] = self.max_y

        transformed_perfs = np.array(transformed_perfs, dtype=np.float64)
        transformed_perfs = get_transform_function(transform)(transformed_perfs)
        return transformed_perfs

    def get_transformed_constraint_perfs(self, bilog_transform=True):
        def bilog(y: np.ndarray):
            """Magnify the difference between y and 0"""
            idx = (y >= 0)
            y_copy = y.copy()
            y_copy[idx] = np.log(1 + y_copy[idx])
            y_copy[~idx] = -np.log(1 - y_copy[~idx])
            return y_copy

        if self.num_constraints == 0:
            return None

        transformed_constraint_perfs = self.constraint_perfs.copy()
        success_constraint_perfs = [c for c in transformed_constraint_perfs if c is not None]
        max_c = np.max(success_constraint_perfs, axis=0) if success_constraint_perfs else [1.0] * self.num_constraints
        for i in self.failed_index:
            transformed_constraint_perfs[i] = max_c

        transformed_constraint_perfs = np.array(transformed_constraint_perfs, dtype=np.float64)
        if bilog_transform:
            transformed_constraint_perfs = bilog(transformed_constraint_perfs)
        return transformed_constraint_perfs

    def get_internal_metrics(self):
        return self.internal_metrics

    def get_perf(self, config: Configuration):
        return self.data[config]

    def get_all_perfs(self):
        return list(self.data.values())

    def get_all_configs(self):
        return list(self.data.keys())

    def empty(self):
        return self.config_counter == 0

    def get_incumbents(self):
        return self.incumbents

    def save_json(self, fn: str = "history_container.json"):
        data = []
        for i in range(len(self.perfs)):
            tmp = {
                'configuration': self.configurations_all[i].get_dictionary(),
                'external_metrics': self.external_metrics[i],
                'internal_metrics': self.internal_metrics[i],
                'resource': self.resource[i],
                'context': self.contexts[i],
                'trial_state': self.trial_states[i],
                'elapsed_time': self.elapsed_times[i],
                'iter_time': self.iter_times[i]
            }
            data.append(tmp)

        with open(fn, "w") as fp:
            json.dump({"info": self.info,  "data": data}, fp, indent=2)

    def load_history_from_json(self, fn: str = "history_container.json", load_num=None):  # todo: all configs
        try:
            with open(fn) as fp:
                all_data = json.load(fp)
        except Exception as e:
            self.logger.warning(
                'Encountered exception %s while reading runhistory from %s. '
                'Not adding any runs!', e, fn,
            )
            return

        info = all_data["info"]
        data = all_data["data"]

        y_variables = info['objs']
        c_variables = info['constraints']
        self.num_constraints = len(c_variables)
        self.info = info

        assert len(self.info['objs']) == self.num_objs
        assert len(self.info['constraints']) == self.num_constraints
        knobs_target = self.config_space.get_hyperparameter_names()
        knobs_default = self.config_space.get_default_configuration().get_dictionary()

        if not load_num is None:
            data = data[:load_num]
        for tmp in data:
            config_dict = tmp['configuration'].copy()
            knobs_source = tmp['configuration'].keys()
            knobs_delete = [knob for knob in knobs_source if knob not in knobs_target]
            knobs_add = [knob for knob in knobs_target if knob not in knobs_source]

            for knob in knobs_delete:
                config_dict.pop(knob)
            for knob in knobs_add:
                config_dict[knob] = knobs_default[knob]

            config = Configuration(self.config_space, config_dict)
            em = tmp['external_metrics']
            im = tmp['internal_metrics']
            resource = tmp['resource']
            trial_state = tmp['trial_state']
            elapsed_time = tmp['elapsed_time']
            iter_time = tmp['iter_time'] if 'iter_time' in tmp.keys() else tmp['elapsed_time']
            context = tmp['context'] if 'context' in tmp.keys() else None
            res = dict(em, **resource)

            self.configurations.append(config)
            self.configurations_all.append(self.fill_default_value(config))
            self.trial_states.append(trial_state)
            self.elapsed_times.append(elapsed_time)
            self.iter_times.append(iter_time)
            self.internal_metrics.append(im)
            self.external_metrics.append(em)
            self.resource.append(resource)
            self.contexts.append(context)

            objs = self.get_objs(res, y_variables)
            if self.num_objs == 1:
                self.perfs.append(objs[0])
            else:
                self.perfs.append(objs)

            constraints = self.get_constraints(res, c_variables)
            self.constraint_perfs.append(constraints)

            transform_perf = False
            failed = False
            if trial_state == SUCCESS and all(perf < MAXINT for perf in objs):
                if self.num_constraints > 0 and constraints is None:
                    self.logger.error('Constraint is None in a SUCCESS trial!')
                    failed = True
                    transform_perf = True
                else:
                    # If infeasible, transform perf to the largest found objective value
                    feasible = True
                    if self.num_constraints > 0 and any(c > 0 for c in constraints):
                        transform_perf = True
                        feasible = False

                    if self.num_objs == 1:
                        self.successful_perfs.append(objs[0])
                    else:
                        self.successful_perfs.append(objs)
                    if feasible:
                        if self.num_objs == 1:
                            self.add(config, objs[0])
                        else:
                            self.add(config, objs)
                    else:
                        self.add(config, MAXINT)

                    self.perc = np.percentile(self.successful_perfs, self.scale_perc, axis=0)
                    self.min_y = np.min(self.successful_perfs, axis=0).tolist()
                    self.max_y = np.max(self.successful_perfs, axis=0).tolist()

            else:
                # failed trial
                failed = True
                transform_perf = True

            cur_idx = len(self.perfs) - 1
            if transform_perf:
                self.transform_perf_index.append(cur_idx)
            if failed:
                self.failed_index.append(cur_idx)

    def get_objs(self, res, y_variables):
        try:
            objs = []
            for y_variable in y_variables:
                key = y_variable.strip().strip('-')
                value = res[key]
                if not y_variable.strip()[0] == '-':
                    value = - value
                objs.append(value)
        except:
            objs = [MAXINT] * self.num_objs

        return objs

    def get_constraints(self, res, constraints):
        if len(constraints) == 0:
            return None

        try:
            locals().update(res)
            constraintL = []
            for constraint in constraints:
                value = eval(constraint)
                constraintL.append(value)
        except:
            constraintL = []

        return constraintL

    def alter_configuration_space(self, new_space: ConfigurationSpace):
        names = new_space.get_hyperparameter_names()
        all_default_config_dict = self.config_space_all.get_default_configuration().get_dictionary()

        configurations = []
        data = collections.OrderedDict()

        for i in range(len(self.configurations)):
            config = self.configurations_all[i]
            config_new = {}
            for name in names:
                if name in config.get_dictionary().keys():
                    config_new[name] = config[name]
                else:
                    config_new[name] = all_default_config_dict[name]

            c_new = Configuration(new_space, config_new)
            configurations.append(c_new)
            perf = self.perfs[i]
            data[c_new] = perf

        self.configurations = configurations
        self.data = data
        self.config_space = new_space

    def get_str(self):
        from terminaltables import AsciiTable
        incumbents = self.get_incumbents()
        if not incumbents:
            return 'No incumbents in history. Please run optimization process.'

        configs_table = []
        nil = "-"
        parameters = list(incumbents[0][0].get_dictionary().keys())
        for para in parameters:
            row = []
            row.append(para)
            for config, perf in incumbents:
                val = config.get(para, None)
                if val is None:
                    val = nil
                if isinstance(val, float):
                    val = "%.6f" % val
                elif not isinstance(val, str):
                    val = str(val)
                row.append(val)
            configs_table.append(row)
        configs_title = ["Parameters"] + ["" if i else "Optimal Value" for i, _ in enumerate(incumbents)]

        table_data = ([configs_title] +
                      configs_table +
                      [["Optimal Objective Value"] + [perf for config, perf in incumbents]] +
                      [["Num Configs"] + [str(len(self.configurations))]]
                      )

        M = 2
        raw_table = AsciiTable(
            table_data
            # title="Result of Optimization"
        ).table
        lines = raw_table.splitlines()
        title_line = lines[1]
        st = title_line.index("|", 1)
        col = "Optimal Value"
        L = len(title_line)
        lines[0] = "+" + "-" * (L - 2) + "+"
        new_title_line = title_line[:st + 1] + (" " + col + " " * (L - st - 3 - len(col))) + "|"
        lines[1] = new_title_line
        bar = "\n" + lines.pop() + "\n"
        finals = lines[-M:]
        prevs = lines[:-M]
        render_table = "\n".join(prevs) + bar + bar.join(finals) + bar
        return render_table

    def __str__(self):
        return self.get_str()

    __repr__ = __str__

    def visualize_jupyter(self):
        try:
            import hiplot as hip
        except ModuleNotFoundError:
            if sys.version_info < (3, 6):
                raise ValueError("HiPlot requires Python 3.6 or newer. "
                                 "See https://facebookresearch.github.io/hiplot/getting_started.html")
            self.logger.error("Please run 'pip install hiplot'. "
                              "HiPlot requires Python 3.6 or newer.")
            raise

        visualize_data = []
        for config, perf in zip(self.configurations, self.perfs):
            config_perf = config.get_dictionary().copy()
            assert 'perf' not in config_perf.keys()
            config_perf['perf'] = perf
            visualize_data.append(config_perf)
        hip.Experiment.from_iterable(visualize_data).display()
        return

    def get_importance(self, config_space=None, return_list=False):
        def _get_X(configurations, config_space):
            from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter, Constant
            X_from_dict = np.array([list(config.get_dictionary().values()) for config in configurations])
            X_from_array = np.array([config.get_array() for config in configurations])
            discrete_types = (CategoricalHyperparameter, OrdinalHyperparameter, Constant)
            discrete_idx = [isinstance(hp, discrete_types) for hp in config_space.get_hyperparameters()]
            X = X_from_dict.copy()
            X[:, discrete_idx] = X_from_array[:, discrete_idx]
            return X

        try:
            import pyrfr.regression as reg
            import pyrfr.util
        except ModuleNotFoundError:
            self.logger.error(
                'To use get_importance(), please install pyrfr: '
                'https://open-box.readthedocs.io/en/latest/installation/install_pyrfr.html'
            )
            raise
        from autotune.utils.fanova import fANOVA
        from terminaltables import AsciiTable

        if config_space is None:
            config_space = self.config_space
        if config_space is None:
            raise ValueError('Please provide config_space to show parameter importance!')

        X = _get_X(self.configurations, config_space)
        Y = np.array(self.get_transformed_perfs())

        # create an instance of fanova with data for the random forest and the configSpace
        f = fANOVA(X=X, Y=Y, config_space=config_space)

        # marginal for first parameter
        keys = [hp.name for hp in config_space.get_hyperparameters()]
        importance_list = list()
        for key in keys:
            p_list = (key,)
            res = f.quantify_importance(p_list)
            individual_importance = res[(key,)]['individual importance']
            importance_list.append([key, individual_importance])
        importance_list.sort(key=lambda x: x[1], reverse=True)

        if return_list:
            return importance_list

        for item in importance_list:
            item[1] = '%.6f' % item[1]
        table_data = [["Parameters", "Importance"]] + importance_list
        importance_table = AsciiTable(table_data).table
        return importance_table

    def get_shap_importance(self, config_space=None, return_dir=False, config_bench=None):
        import shap
        from lightgbm import LGBMRegressor
        from terminaltables import AsciiTable

        if config_space is None:
            config_space = self.config_space
        if config_space is None:
            raise ValueError('Please provide config_space to show parameter importance!')

        if config_bench is None:
            X_bench = self.config_space.get_default_configuration().get_array().reshape(1,-1)
        else:
            X_bench = config_bench.get_array().reshape(1,-1)

        X = np.array([list(config.get_array()) for config in self.configurations])
        Y = -  np.array(self.get_transformed_perfs())

        # Fit a LightGBMRegressor with observations
        lgbr = LGBMRegressor()
        lgbr.fit(X, Y)
        explainer = shap.TreeExplainer(lgbr)
        X_selected = X[Y>=self.get_default_performance()]
        if X_selected.shape[0] == 0:
            X_selected = X[Y >= np.quantile(Y, 0.9)]

        shap_values = explainer.shap_values(X)
        #shap_value_default = explainer.shap_values(X_bench)[-1]
        delta = shap_values #- shap_value_default
        delta = np.where(delta > 0, delta, 0)
        feature_importance = np.average(delta, axis=0)

        keys = [hp.name for hp in config_space.get_hyperparameters()]
        importance_list = []
        for i, hp_name in enumerate(keys):
            importance_list.append([hp_name, feature_importance[i]])
        importance_list.sort(key=lambda x: x[1], reverse=True)

        importance_dir = dict()
        for item in importance_list:
            importance_dir[item[0]] = item[1]

        if return_dir:
            return importance_dir

        for item in importance_list:
            item[1] = '%.6f' % item[1]
        table_data = [["Parameters", "Importance"]] + importance_list
        importance_table = AsciiTable(table_data).table

        return importance_table

    def plot_convergence(
            self,
            xlabel="Number of iterations $n$",
            ylabel=r"Min objective value after $n$ iterations",
            ax=None, name=None, alpha=0.2, yscale=None,
            color=None, true_minimum=None,
            **kwargs):
        """Plot one or several convergence traces.

        Parameters
        ----------
        args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
            The result(s) for which to plot the convergence trace.

            - if `OptimizeResult`, then draw the corresponding single trace;
            - if list of `OptimizeResult`, then draw the corresponding convergence
              traces in transparency, along with the average convergence trace;
            - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
              an `OptimizeResult` or a list of `OptimizeResult`.

        ax : `Axes`, optional
            The matplotlib axes on which to draw the plot, or `None` to create
            a new one.

        true_minimum : float, optional
            The true minimum value of the function, if known.

        yscale : None or string, optional
            The scale for the y-axis.

        Returns
        -------
        ax : `Axes`
            The matplotlib axes.
        """
        losses = list(self.perfs)

        n_calls = len(losses)
        iterations = range(1, n_calls + 1)
        mins = [np.min(losses[:i]) for i in iterations]
        max_mins = max(mins)
        cliped_losses = np.clip(losses, None, max_mins)
        return plot_convergence(iterations, mins, cliped_losses, xlabel, ylabel, ax, name, alpha, yscale, color,
                                true_minimum, **kwargs)


    def get_default_performance(self):
        default_array = self.config_space.get_default_configuration().get_array()
        default_list = list()
        for i,config in enumerate(self.configurations):
            if (config.get_array() == default_array).all():
                default_list.append(self.get_transformed_perfs()[i])

        if not len(default_list):
            return self.get_transformed_perfs()[0]
        else:
            return  sum(default_list)/len(default_list)

    def get_promising_space(self, quantile_threshold=0, respect=False):
        y = - self.get_transformed_perfs()
        if quantile_threshold == 0:
            performance_threshold = - self.get_default_performance()
        else:
            performance_threshold = np.quantile(y, quantile_threshold)
            if performance_threshold <  - self.get_default_performance() and not respect:
                performance_threshold = - self.get_default_performance()

        X = convert_configurations_to_array(self.configurations)

        X_bad = X[y<performance_threshold]
        X_good = X[y>=performance_threshold]
        pruned_space = dict()
        importances = self.get_shap_importance(return_dir=True)

        for j in range(X.shape[1]):
            config = self.config_space.get_hyperparameter_names()[j]
            if isinstance(self.config_space.get_hyperparameters()[j], CategoricalHyperparameter ):
                good_values = np.unique(X_good[:, j])
                true_values = list()
                for t in range(good_values.shape[0]):
                    value = good_values[t]
                    true_value = Configuration(self.config_space, vector=X[X[:, j] == value][0])[config]
                    true_values.append(true_value)

                pruned_space[config] = (true_values, None, importances[config] / abs(self.get_default_performance()))
                continue

            p_good_max = X_good[:, j].max()
            lager_set = X_bad[:, j][X_bad[:, j] > p_good_max]
            if not lager_set.shape[0]:
                p_max = p_good_max
            else:
                p_max = min(lager_set)
            p_good_min = X_good[:, j].min()
            smaller_set = X_bad[:, j][X_bad[:, j] < p_good_min]
            if not smaller_set.shape[0]:
                p_min = p_good_min
            else:
                p_min = max(smaller_set)
            pruned_space[config] = (p_min, p_max, importances[config] / abs(self.get_default_performance()))

        return pruned_space

class MOHistoryContainer(HistoryContainer):
    """
    Multi-Objective History Container
    """

    def __init__(self, task_id, num_objs, num_constraints=0, config_space=None, ref_point=None):
        super().__init__(task_id=task_id, num_constraints=num_constraints, config_space=config_space)
        self.pareto = collections.OrderedDict()
        self.num_objs = num_objs
        self.mo_incumbent_value = [MAXINT] * self.num_objs
        self.mo_incumbents = [list() for _ in range(self.num_objs)]
        self.ref_point = ref_point
        self.hv_data = list()

        self.max_y = [MAXINT] * self.num_objs

    def add(self, config: Configuration, perf: List[Perf]):
        assert self.num_objs == len(perf)

        if config in self.data:
            self.logger.warning('Repeated configuration detected!')
            return

        self.data[config] = perf
        self.data_all[self.fill_default_value(config)] = perf
        self.config_counter += 1

        # update pareto
        remove_config = []
        for pareto_config, pareto_perf in self.pareto.items():  # todo efficient way?
            if all(pp <= p for pp, p in zip(pareto_perf, perf)):
                break
            elif all(p <= pp for pp, p in zip(pareto_perf, perf)):
                remove_config.append(pareto_config)
        else:
            self.pareto[config] = perf
            self.logger.info('Update pareto: config=%s, objs=%s.' % (str(config), str(perf)))

        for conf in remove_config:
            self.logger.info('Remove from pareto: config=%s, objs=%s.' % (str(conf), str(self.pareto[conf])))
            self.pareto.pop(conf)

        # update mo_incumbents
        for i in range(self.num_objs):
            if len(self.mo_incumbents[i]) > 0:
                if perf[i] < self.mo_incumbent_value[i]:
                    self.mo_incumbents[i].clear()
                if perf[i] <= self.mo_incumbent_value[i]:
                    self.mo_incumbents[i].append((config, perf[i], perf))
                    self.mo_incumbent_value[i] = perf[i]
            else:
                self.mo_incumbent_value[i] = perf[i]
                self.mo_incumbents[i].append((config, perf[i], perf))

        # Calculate current hypervolume if reference point is provided
        if self.ref_point is not None:
            pareto_front = self.get_pareto_front()
            if pareto_front:
                hv = Hypervolume(ref_point=self.ref_point).compute(pareto_front)
            else:
                hv = 0
            self.hv_data.append(hv)

    def get_incumbents(self):
        return self.get_pareto()

    def get_mo_incumbents(self):
        return self.mo_incumbents

    def get_mo_incumbent_value(self):
        return self.mo_incumbent_value

    def get_pareto(self):
        return list(self.pareto.items())

    def get_pareto_set(self):
        return list(self.pareto.keys())

    def get_pareto_front(self):
        return list(self.pareto.values())

    def compute_hypervolume(self, ref_point=None):
        if ref_point is None:
            ref_point = self.ref_point
        assert ref_point is not None
        pareto_front = self.get_pareto_front()
        if pareto_front:
            hv = Hypervolume(ref_point=ref_point).compute(pareto_front)
        else:
            hv = 0
        return hv

    def plot_convergence(self, *args, **kwargs):
        raise NotImplementedError('plot_convergence only supports single objective!')

    def visualize_jupyter(self, *args, **kwargs):
        raise NotImplementedError('visualize_jupyter only supports single objective!')




class MultiStartHistoryContainer(object):
    """
    History container for multistart algorithms.
    """

    def __init__(self, task_id, num_objs=1, num_constraints=0, ref_point=None):
        self.task_id = task_id
        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.history_containers = []
        self.ref_point = ref_point
        self.current = None
        self.restart()

    def restart(self):
        if self.num_objs == 1:
            self.current = HistoryContainer(self.task_id, self.num_constraints)
        else:
            self.current = MOHistoryContainer(self.task_id, self.num_objs, self.num_constraints, self.ref_point)
        self.history_containers.append(self.current)

    def get_configs_for_all_restarts(self):
        all_configs = []
        for history_container in self.history_containers:
            all_configs.extend(list(history_container.data.keys()))
        return all_configs

    def get_incumbents_for_all_restarts(self):
        best_incumbents = []
        best_incumbent_value = float('inf')
        if self.num_objs == 1:
            for hc in self.history_containers:
                incumbents = hc.get_incumbents()
                incumbent_value = hc.incumbent_value
                if incumbent_value > best_incumbent_value:
                    continue
                elif incumbent_value < best_incumbent_value:
                    best_incumbent_value = incumbent_value
                best_incumbents.extend(incumbents)
            return best_incumbents
        else:
            return self.get_pareto_front()

    def get_pareto_front(self):
        assert self.num_objs > 1
        Y = np.vstack([hc.get_pareto_front() for hc in self.history_containers])
        return get_pareto_front(Y).tolist()

    def update_observation(self, observation: Observation):
        return self.current.update_observation(observation)

    def add(self, config: Configuration, perf: Perf):
        self.current.add(config, perf)

    @property
    def configurations(self):
        return self.current.configurations

    @property
    def perfs(self):
        return self.current.perfs

    @property
    def constraint_perfs(self):
        return self.current.constraint_perfs

    @property
    def trial_states(self):
        return self.current.trial_states

    @property
    def successful_perfs(self):
        return self.current.successful_perfs

    def get_transformed_perfs(self):
        return self.current.get_transformed_perfs

    def get_transformed_constraint_perfs(self):
        return self.current.get_transformed_constraint_perfs

    def get_perf(self, config: Configuration):
        for history_container in self.history_containers:
            if config in history_container.data:
                return self.data[config]
        raise KeyError

    def get_all_configs(self):
        return self.current.get_all_configs()

    def empty(self):
        return self.current.config_counter == 0

    def get_incumbents(self):
        if self.num_objs == 1:
            return self.current.incumbents
        else:
            return self.current.get_pareto()

    def get_mo_incumbents(self):
        assert self.num_objs > 1
        return self.current.mo_incumbents

    def get_mo_incumbent_value(self):
        assert self.num_objs > 1
        return self.current.mo_incumbent_value

    def get_pareto(self):
        assert self.num_objs > 1
        return self.current.get_pareto()

    def get_pareto_set(self):
        assert self.num_objs > 1
        return self.current.get_pareto_set()

    def compute_hypervolume(self, ref_point=None):
        assert self.num_objs > 1
        return self.current.compute_hypervolume(ref_point)

    def save_json(self, fn: str = "history_container.json"):
        """
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        """
        self.current.save_json(fn)

    def load_history_from_json(self, cs: ConfigurationSpace, fn: str = "history_container.json"):
        """Load and runhistory in json representation from disk.
        Parameters
        ----------
        fn : str
            file name to load from
        cs : ConfigSpace
            instance of configuration space
        """
        self.current.load_history_from_json(cs, fn)

