import pickle
import numpy as np
import os
import glob
import json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from reward_model import GPT_wrapper
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import gc
from scipy.stats import ttest_ind
from pathlib import Path
import random

random.seed(0)
np.random.seed(0)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_results(paths, verbose=False):
    '''
        paths : path to directory containing experiment trials
    '''
    scores = []
    for i, path in enumerate(sorted(paths)):
        score = load_result(path)
        if verbose:
            print(path, score)
        if score is None:
            continue
        scores.append(score)

    if len(scores) > 0:
        mean = np.mean(scores)
    else:
        mean = np.nan

    if len(scores) > 1:
        err = np.std(scores) / np.sqrt(len(scores))
    else:
        err = 0
    return mean, err, scores


def load_result(path):
    '''
        path : path to experiment directory; expects `rollout.json` to be in directory
    '''
    fullpath = os.path.join(path, 'rollout.json')

    if not os.path.exists(fullpath):
        return None

    results = json.load(open(fullpath, 'rb'))
    score = results['score'] * 100
    return score


def compute_cp_reward(input_traj='', verbose=False, alg='', desired_num_samples=1000, diffusion=True):
    info = {}
    num_samples = len(os.listdir(input_traj))
    all_samples = []

    for sample_i in range(num_samples):
        sample_name = os.path.join(input_traj, '{}'.format(sample_i), 'rollout.json')
        if not os.path.exists(sample_name):
            continue
        results = json.load(open(sample_name, 'rb'))
        score = results['score'] * 100

        all_samples.append(np.asarray([score])[:, None, None])
    all_samples_array = np.concatenate(all_samples, axis=0)
    num_repeat = desired_num_samples // all_samples_array.shape[0] + 1
    all_samples_list = [all_samples_array] * num_repeat
    all_samples_array = np.concatenate(all_samples_list, axis=0)[:desired_num_samples]
    all_samples_array = all_samples_array
    fig = plt.figure()
    print('all_samples_array.shape: ', all_samples_array.shape)
    df = pd.DataFrame(all_samples_array.squeeze(-1), columns=["reward"])
    if alg == 'diffuser':
        color = sns.xkcd_rgb['red']
        sns.histplot(data=df, x="reward", binwidth=1, color=color)
    else:
        color = sns.xkcd_rgb['blue']
        sns.histplot(data=df, x="reward", binwidth=1, color=color)
    fig_save_path = os.path.join(input_traj + Path(input_traj).parts[-3] + alg + 'score_histogram_plt.png'.format())
    print('Save to {}'.format(fig_save_path))
    fig.savefig(fig_save_path)
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')
    plt.close(fig)
    gc.collect()

    calibration_set = all_samples_array[:int(all_samples_array.shape[0] * 0.2)]
    test_set = all_samples_array[int(all_samples_array.shape[0] * 0.2):]

    # Get scores
    cal_labels = calibration_set
    model_upper = calibration_set.max(0, keepdims=True)  # [None, :]
    model_lower = calibration_set.min(0, keepdims=True)  # [None, :]
    print('calibration_set.shape: ', calibration_set.shape, 'model_upper: ', model_upper.shape)
    cal_scores = cal_labels  # np.maximum(model_upper - cal_labels, cal_labels - model_lower)
    print('cal_scores.shape: ', cal_scores.shape)
    cal_scores = cal_scores.mean(-2, keepdims=True).mean(-1, keepdims=True)
    n = calibration_set.shape[0]
    print('n: ', n)  # n: 20 # 16000
    alpha = 0.1  # 1 - alpha is the desired coverage
    qhat = np.quantile(cal_scores,
                       min(np.ceil((n + 1) * (1 - alpha)) / n, np.asarray([1])),
                       keepdims=True,
                       interpolation='higher')
    small_qhat = np.quantile(cal_scores,
                       min(np.ceil((n + 1) * (alpha)) / n, np.asarray([1])),
                       keepdims=True,
                       interpolation='higher')
    calibration_set_uncertainty = 2 * qhat

    # Deploy (output=lower and upper adjusted quantiles)
    info['calibration_set_uncertainty'] = calibration_set_uncertainty  # .detach()
    prediction_set = [test_set - qhat, test_set + qhat]
    info['prediction_set'] = prediction_set  # .detach()
    info['test_set'] = test_set
    reward_mean = np.mean(all_samples_array)
    reward_std = np.std(all_samples_array) # / len(all_samples_array) ** 0.5
    total_num_samples = 0
    num_samples_in_cp = 0
    for each_test_set in test_set:
        total_num_samples = total_num_samples + 1
        if small_qhat < each_test_set < qhat:
            num_samples_in_cp = num_samples_in_cp + 1
    validity = num_samples_in_cp / total_num_samples * 100
    interval = qhat - small_qhat
    res_p_value = ttest_ind(calibration_set, test_set)
    return [reward_mean, reward_std, small_qhat, qhat, interval, validity, res_p_value]



if __name__ == "__main__":
    input_traj_list = ['/home/jsun/Programs/UAMBRL/packages/diffuser_maze2d/logs_diffuser/maze2d-umaze-v1/plans_conditional_False/release_H128_T64_LimitsNormalizer_b1_condFalse',
                       '/home/jsun/Programs/UAMBRL/packages/diffuser_maze2d/logs_diffuser/maze2d-medium-v1/plans_conditional_False/release_H256_T256_LimitsNormalizer_b1_condFalse',
                       '/home/jsun/Programs/UAMBRL/packages/diffuser_maze2d/logs_diffuser/maze2d-large-v1/plans_conditional_False/release_H384_T256_LimitsNormalizer_b1_condFalse',
                       ]

    all_mean_list = []
    for each_input_traj in input_traj_list:
        qhat = compute_cp_reward(each_input_traj, alg='diffuser')
        print(
            'each_input_traj: {}, reward_mean: {}, reward_std: {}, small_qhat: {}, qhat: {}, interval: {}, validity: {}, res_p_value: {}'
            .format(each_input_traj, qhat[0], qhat[1], qhat[2], qhat[3], qhat[4], qhat[5], qhat[6]))
        all_mean_list.append(qhat[0])
    print('all reward mean: {}'.format(np.mean(all_mean_list)))
