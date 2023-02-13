import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import copy
import pdb

from autotune.utils.constants import MAXINT, SUCCESS

methodL = ['SMAC', 'MBO', 'DDPG', 'GA']
workloadL = ['twitter', 'sysbench', 'job', 'tpch']
spaceL = [197, 100, 50, 25, 12, 6]


def get_objs(res, y_variables):
    try:
        objs = []
        for y_variable in y_variables:
            key = y_variable.strip().strip('-')
            value = res[key]
            if not y_variable.strip()[0] == '-':
                value = - value
            objs.append(value)
    except:
        objs = [MAXINT]

    return objs[0]


def parse_data_onefile(fn):
    try:
        with open(fn) as fp:
            all_data = json.load(fp)
    except Exception as e:
        print('Encountered exception %s while reading runhistory from %s. '
              'Not adding any runs!', e, fn, )
        return

    info = all_data["info"]
    data = all_data["data"]
    y_variables = info['objs']
    objs, bests, configs = list(), list(), list()
    for tmp in data:
        em = tmp['external_metrics']
        resource = tmp['resource']
        res = dict(em, **resource)
        obj = get_objs(res, y_variables)
        objs.append(obj)
        config = tmp['configuration']
        configs.append(config)
        if not len(bests) or obj < bests[-1]:
            bests.append(obj)
        else:
            bests.append(bests[-1])
    objs_remove_maxint = [item for item in objs if item < MAXINT]
    objs_lagest = max(objs_remove_maxint)
    objs_scaled = list()
    for obj in objs:
        if obj < MAXINT:
            objs_scaled.append(obj)
        else:
            objs.append(objs_lagest)

    return objs_scaled, bests, configs


def parse_data(file_dict):
    comparison = {}
    for method in file_dict.keys():
        comparison[method] = list()
        for file in file_dict[method]:
            objs, bests, _ = parse_data_onefile(file)
            comparison[method].append({
                'objs': objs,
                'bests': bests,
                'n_calls': len(objs)
            })
    return comparison


def get_best(file, iter):
    objs, bests, _ = parse_data_onefile(file)
    if len(bests) < iter:
        print("{} only has {} record, but require {}".format(file, len(bests), iter))
        iter = len(bests)

    return bests[iter - 1]


def plot_comparison(file_dict, workload, figname='plot/tmp.png', **kwargs):
    comparison = parse_data(file_dict)
    ax = plt.gca()
    ax.set_title("{} Convergence plot".format(workload))
    ax.set_xlabel(r"Number of iterations $n$")
    ax.grid()

    for i, method in enumerate(comparison.keys()):
        plot_scatter = False
        all_data = comparison[method]
        n_calls = np.max([_['n_calls'] for _ in all_data])
        bests = []
        if len(all_data) == 1:
            plot_scatter = True

        for data in all_data:
            if len(data['bests']) < n_calls:
                print("{}th file for {} lacks {} record".format(all_data.index(data), method,
                                                                n_calls - len(data['best'])))
            while len(data['bests']) < n_calls:
                data['bests'].append(data['bests'][-1])
            bests.append(data['bests'])

        iterations = range(1, int(n_calls) + 1)
        df_best_y = pd.DataFrame(bests)
        if bests[0][0] < 0:
            ax.set_ylabel(r"Throughput (txn/sec)")
            df_best_y = - df_best_y
            data['objs'] = [-item for item in data['objs']]
        else:
            ax.set_ylabel(r"95th latency (sec)")

        min = df_best_y.quantile(0.25)
        max = df_best_y.quantile(0.75)
        mean = df_best_y.mean()

        # ax.plot(iterations, mean, color=tab10[i], label=method,  **kwargs)
        ax.plot(iterations, mean, color="C{}".format(i), label=method, **kwargs)
        ax.fill_between(iterations, min, max, alpha=0.2, color="C{}".format(i))
        # if plot_scatter:
        #    ax.scatter(iterations, data['objs'], color="C{}".format(i), )

    # plt.ylim(130, 160)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def plot_method(file_dict, workload):
    for method in methodL:
        file_dict_ = copy.deepcopy(file_dict)
        for key in file_dict.keys():
            if not method.upper() in key:
                del file_dict_[key]

        plot_comparison(file_dict_, workload, figname='plot/{}_{}.png'.format(workload,method))


def get_rank(file_dict, keyword, iter=200):
    file_dict_ = copy.deepcopy(file_dict)
    for key in file_dict.keys():
        if not str(keyword) in key and not str(keyword) in file_dict_[key][0]:
            del file_dict_[key]

    performance = np.zeros(4)
    for key in file_dict_.keys():
        per = get_best(file_dict_[key][0], iter)
        method = key.split('-')[1]
        performance[methodL.index(method)] = per

    return performance.argsort().argsort()


def plot_rank(file_dict, type='one-workload', keyword='', iter=200, figname='plot/tmp.png'):
    rankL = list()
    if not keyword == '':
        file_dict_ = copy.deepcopy(file_dict)
        for key in file_dict.keys():
            if not keyword in key:
                del file_dict_[key]


    if type == 'one-workload':
        filterL = spaceL
    elif type == 'one-space':
        filterL = workloadL
    else:
        filterL1 = workloadL
        filterL2 = spaceL

    if type in ['one-workload', 'one-space' ]:
        for item in filterL:
            rank = get_rank(file_dict_, item, iter)
            rankL.append(rank)
            print(rank)
    else:
        for item1 in filterL1:
            file_dict_ = copy.deepcopy(file_dict)
            for key in file_dict.keys():
                if not item1 in key:
                    del file_dict_[key]
            for item2 in filterL2:
                rank = get_rank(file_dict_, item2, iter)
                rankL.append(rank)
                print(rank)

    rankL_T = np.array(rankL).T.tolist()
    ax = plt.gca()
    ax.grid()
    ax.boxplot(rankL_T)
    plt.title(keyword)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(methodL)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


if __name__ == '__main__':
    history_path = '/data2/ruike/DBTune/scripts/DBTune_history/'

    #######90:Twitter#################
    file_dict_twitter = {
        'SMAC-197': [os.path.join(history_path, 'history_sysbench_smac_197_2.json')],
        'MBO-197': [os.path.join(history_path, 'history_twitter_mbo_197.json')],
        'GA-197': [os.path.join(history_path, 'history_twitter_ga_197.json')],
        'DDPG-197': [os.path.join(history_path, 'history_twitter_ddpg_197.json')],
        'MBO-100': [os.path.join(history_path, 'history_twitter_mbo_100.json')],
        'SMAC-100': [os.path.join(history_path, 'history_twitter_smac_100.json')],
        'DDPG-100': [os.path.join(history_path, 'history_twitter_ddpg_100.json')],
        'GA-100': [os.path.join(history_path, 'history_twitter_ga_100.json')],
        'MBO-50': [os.path.join(history_path, 'history_twitter_mbo_50.json')],
        'SMAC-50': [os.path.join(history_path, 'history_twitter_smac_50.json')],
        'GA-50': [os.path.join(history_path, 'history_twitter_ga_50.json')],
        'DDPG-50': [os.path.join(history_path, 'history_twitter_ddpg_50.json')],
        'MBO-25': [os.path.join(history_path, 'history_twitter_mbo_25.json')],
        'SMAC-25': [os.path.join(history_path, 'history_twitter_smac_25.json')],
        'DDPG-25': [os.path.join(history_path, 'history_twitter_ddpg_25.json')],
        'GA-25': [os.path.join(history_path, 'history_twitter_ga_25.json')],
        'MBO-12': [os.path.join(history_path, 'history_twitter_mbo_12.json')],
        'SMAC-12': [os.path.join(history_path, 'history_twitter_smac_12.json')],
        'DDPG-12': [os.path.join(history_path, 'history_twitter_ddpg_12.json')],
        'GA-12': [os.path.join(history_path, 'history_twitter_ga_12.json')],
        'MBO-6': [os.path.join(history_path, 'history_twitter_mbo_6.json')],
        'SMAC-6': [os.path.join(history_path, 'history_twitter_smac_6.json')],
        'DDPG-6': [os.path.join(history_path, 'history_twitter_ddpg_6.json')],
        'GA-6': [os.path.join(history_path, 'history_twitter_ga_6.json')],
    }

    # get_rank(file_dict, iter=200)
    # plot_comparison(file_dict_twitter, 'Twitter')
    # plot_method(file_dict_twitter, 'Twitter')

    #######172:SYSBENCH#################
    file_dict_sysbench = {'SMAC-197': [os.path.join(history_path, 'history_sysbench_smac_197.json')],
                          'MBO-197': [os.path.join(history_path, 'history_sysbench_mbo_197.json')],
                          'GA-197': [os.path.join(history_path, 'history_sysbench_ga_197.json')],
                          'DDPG-197': [os.path.join(history_path, 'history_sysbench_ddpg_197.json')],
                          'MBO-100': [os.path.join(history_path, 'history_sysbench_mbo_100.json')],
                          'SMAC-100': [os.path.join(history_path, 'history_sysbench_smac_100.json')],
                          'DDPG-100': [os.path.join(history_path, 'history_sysbench_ddpg_100.json')],
                          'GA-100': [os.path.join(history_path, 'history_sysbench_ga_100.json')],
                          'SMAC-50': [os.path.join(history_path, 'history_sysbench_smac_50.json')],
                          'MBO-50': [os.path.join(history_path, 'history_sysbench_mbo_50.json')],
                          'DDPG-50': [os.path.join(history_path, 'history_sysbench_ddpg_50.json')],
                          'GA-50': [os.path.join(history_path, 'history_sysbench_ga_50.json')],
                          'SMAC-25': [os.path.join(history_path, 'history_sysbench_smac_25.json')],
                          'MBO-25': [os.path.join(history_path, 'history_sysbench_mbo_25.json')],
                          'DDPG-25': [os.path.join(history_path, 'history_sysbench_ddpg_25.json')],
                          'GA-25': [os.path.join(history_path, 'history_sysbench_ga_25.json')],
                          'MBO-12': [os.path.join(history_path, 'history_sysbench_mbo_12.json')],
                          'SMAC-12': [os.path.join(history_path, 'history_sysbench_smac_12.json')],
                          'DDPG-12': [os.path.join(history_path, 'history_sysbench_ddpg_12.json')],
                          'GA-12': [os.path.join(history_path, 'history_sysbench_ga_12.json')],
                          'MBO-6': [os.path.join(history_path, 'history_sysbench_mbo_6.json')],
                          'SMAC-6': [os.path.join(history_path, 'history_sysbench_smac_6.json')],
                          'DDPG-6': [os.path.join(history_path, 'history_sysbench_ddpg_6.json')],
                          'GA-6': [os.path.join(history_path, 'history_sysbench_ga_6.json')],
                          }
    # get_rank(file_dict, iter=200)
    # plot_method(file_dict, 'SYSBENCH')
    # plot_comparison(file_dict, 'SYSBENCH')

    #######15:JOB#################
    file_dict_job = {'SMAC-197': [os.path.join(history_path, 'history_job_smac_197.json')],
                     'MBO-197': [os.path.join(history_path, 'history_job_mbo_197.json')],
                     'GA-197': [os.path.join(history_path, 'history_job_ga_197.json')],
                     'DDPG-197': [os.path.join(history_path, 'history_job_ddpg_197.json')],
                     'SMAC-100': [os.path.join(history_path, 'history_job_smac_100.json')],
                     'MBO-100': [os.path.join(history_path, 'history_job_mbo_100.json')],
                     'GA-100': [os.path.join(history_path, 'history_job_ga_100.json')],
                     'DDPG-100': [os.path.join(history_path, 'history_job_ddpg_100.json')],
                     'SMAC-50': [os.path.join(history_path, 'history_job_smac_50.json')],
                     'MBO-50': [os.path.join(history_path, 'history_job_mbo_50.json')],
                     'DDPG-50': [os.path.join(history_path, 'history_job_ddpg_50.json')],
                     'GA-50': [os.path.join(history_path, 'history_job_ga_50.json')],
                     'SMAC-25': [os.path.join(history_path, 'history_job_smac_25.json')],
                     'MBO-25': [os.path.join(history_path, 'history_job_mbo_25.json')],
                     'DDPG-25': [os.path.join(history_path, 'history_job_ddpg_25.json')],
                     'GA-25': [os.path.join(history_path, 'history_job_ga_25.json')],
                     'MBO-12': [os.path.join(history_path, 'history_job_mbo_12.json')],
                     'SMAC-12': [os.path.join(history_path, 'history_job_smac_12.json')],
                     'DDPG-12': [os.path.join(history_path, 'history_job_ddpg_12.json')],
                     'GA-12': [os.path.join(history_path, 'history_job_ga_12.json')],
                     'MBO-6': [os.path.join(history_path, 'history_job_mbo_6.json')],
                     'SMAC-6': [os.path.join(history_path, 'history_job_smac_6.json')],
                     'DDPG-6': [os.path.join(history_path, 'history_job_ddpg_6.json')],
                     'GA-6': [os.path.join(history_path, 'history_job_ga_6.json')],
                     }
    # plot_comparison(file_dict, 'Job')

    #######93:TPCH#################
    file_dict_tpch = {
        'MBO-197': [os.path.join(history_path, 'history_tpch_mbo_197.json')],
        'SMAC-197': [os.path.join(history_path, 'history_tpch_smac_197.json')],
        'DDPG-197': [os.path.join(history_path, 'history_tpch_ddpg_197.json')],
        'GA-197': [os.path.join(history_path, 'history_tpch_ga_197.json')],
        'MBO-100': [os.path.join(history_path, 'history_tpch_mbo_100.json')],
        'SMAC-100': [os.path.join(history_path, 'history_tpch_smac_100.json')],
        'DDPG-100': [os.path.join(history_path, 'history_tpch_ddpg_100.json')],
        'GA-100': [os.path.join(history_path, 'history_tpch_ga_100.json')],
        'SMAC-50': [os.path.join(history_path, 'history_tpch_smac_50.json')],
        'MBO-50': [os.path.join(history_path, 'history_tpch_mbo_50.json')],
        'DDPG-50': [os.path.join(history_path, 'history_tpch_ddpg_50.json')],
        'GA-50': [os.path.join(history_path, 'history_tpch_ga_50.json')],
        'SMAC-25': [os.path.join(history_path, 'history_tpch_smac_25.json')],
        'MBO-25': [os.path.join(history_path, 'history_tpch_mbo_25.json')],
        'DDPG-25': [os.path.join(history_path, 'history_tpch_ddpg_25.json')],
        'GA-25': [os.path.join(history_path, 'history_tpch_ga_25.json')],
        'MBO-12': [os.path.join(history_path, 'history_tpch_mbo_12.json')],
        'SMAC-12': [os.path.join(history_path, 'history_tpch_smac_12.json')],
        'DDPG-12': [os.path.join(history_path, 'history_tpch_ddpg_12.json')],
        'GA-12': [os.path.join(history_path, 'history_tpch_ga_12.json')],
        'MBO-6': [os.path.join(history_path, 'history_tpch_mbo_6.json')],
        'SMAC-6': [os.path.join(history_path, 'history_tpch_smac_6.json')],
        'DDPG-6': [os.path.join(history_path, 'history_tpch_ddpg_6.json')],
        'GA-6': [os.path.join(history_path, 'history_tpch_ga_6.json')],
    }
    # plot_comparison(file_dict, 'TPCH')

    #ALL(90)#################
    all = [file_dict_twitter, file_dict_sysbench, file_dict_job, file_dict_tpch ]
    file_dict_all = dict()
    for i in range(4):
        for key in all[i].keys():
            file_dict_all[workloadL[i]+'-'+key] = all[i][key]


    # for space in spaceL:
    #     print("space {}".format(space))
    #     plot_rank(file_dict_all, iter=200, type='one-space', keyword=str(space), figname='plot/rank_{}.png'.format(space))
    #
    # for workload in workloadL:
    #     print("workload {}".format(workload))
    #     plot_rank(file_dict_all,  iter=200, type='one-workload', keyword=str(workload), figname='plot/rank_{}.png'.format(workload))

    #plot_rank(file_dict_all, iter=200,  type='all', keyword='', figname='plot/rank_all.png')

    # plot_comparison({'MBO-25': [os.path.join(history_path, 'history_sysbench_mbo_25.json')],
    #     'MBO-20-1': [os.path.join(history_path, 'history_test.json')],
    #     'MBO-12': [os.path.join(history_path, 'history_sysbench_mbo_12.json')],}, 'SYSBENCH')