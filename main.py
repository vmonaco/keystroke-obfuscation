import os
import sys
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

from util import load_data, save_results, load_results, save_fig
from preprocess import preprocess
from obfuscation import obfuscate_keystrokes, mean_lag
from keystroke import extract_keystroke_features
from classify import classification_acc, predictions_smape

SEED = 1234

N_FOLDS = 10
MIXES = ['delay', 'interval']
MIX_PARAMS = {'delay': [50, 100, 200, 500, 1000],
              'interval': [0.1, 0.5, 1.0, 1.5, 2.0]}
DATASETS = ['short_fixed', 'long_fixed', 'long_free']
TARGETS = ['user', 'age', 'gender', 'handedness']


def describe(name):
    """
    Describe the dataset
    """
    df = load_data(name)
    s = df.groupby(level=[0, 1]).size()
    print('Dataset               :', name)
    print('Users                 :', len(s.groupby(level=0)))
    print('Sessions/user         :', s.groupby(level=0).size().mean())
    print('Sample size           :', s.mean(), '+/-', s.std())
    print('Mean pp interval (ms) :',
          df.groupby(level=[0, 1]).apply(lambda x: x['timepress'].diff().dropna().mean()).mean())
    print('Mean duration (ms)    :',
          df.groupby(level=[0, 1]).apply(lambda x: (x['timerelease'] - x['timepress']).mean()).mean())

    for target in TARGETS[1:]:
        s = df.reset_index().groupby([target, 'session']).size().groupby(level=0).size()
        print(target)
        print(s / s.sum())
    return


def extract_features(df):
    def make_features(x):
        return pd.Series({
            'age': x.iloc[0]['age'],
            'gender': x.iloc[0]['gender'],
            'handedness': x.iloc[0]['handedness'],
            'features': extract_keystroke_features(x)
        })

    return df.groupby(level=[0, 1]).apply(make_features)


def acc_figure(name):
    df = load_results(name)

    df = df.set_index(['dataset', 'strategy'])
    fig, axes = plt.subplots(4, 3, sharey=True, squeeze=True, figsize=(6, 5))

    for dataset, col in zip(DATASETS, axes.T):
        for target, ax in zip(TARGETS, col):
            ax.plot(np.r_[0, df.loc[(dataset, 'delay'), 'mean_delta'].values / 1000],
                    np.r_[df.loc[(dataset, 'none'), target].iloc[0], df.loc[(dataset, 'delay'), target]], linewidth=1, label='Delay')

            ax.plot(np.r_[0, df.loc[(dataset, 'interval'), 'mean_delta'].values / 1000],
                    np.r_[df.loc[(dataset, 'none'), target].iloc[0], df.loc[(dataset, 'interval'), target]], linewidth=1, linestyle='--', label='Interval')

            ax.set_ylim(0, 1)

            if dataset == 'short_fixed':
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75])
            else:
                ax.set_xlim(0, 2)
                ax.set_xticks([0, 0.5, 1, 1.5])

    axes[0, 0].set_title('Short fixed-text')
    axes[0, 1].set_title('Long fixed-text')
    axes[0, 2].set_title('Long free-text')

    axes[0, 0].set_ylabel('Identity ACC')
    axes[1, 0].set_ylabel('Age ACC')
    axes[2, 0].set_ylabel('Gender ACC')
    axes[3, 0].set_ylabel('Handedness ACC')

    for i,j in product(range(3), range(3)):
        axes[i,j].set_xticklabels([])

    axes[-1,-1].legend(loc='lower right')

    fig.text(0.5, 0.0, 'Lag (s)', ha='center')
    # fig.text(0.0, 0.5, 'ACC', va='center', rotation='vertical')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15)

    save_fig(name)
    return


if __name__ == '__main__':

    if len(sys.argv) > 2:
        print('Usage: python main.py [seed]')
        sys.exit(1)

    if len(sys.argv) == 2:
        seed = int(sys.argv[1])
    else:
        seed = SEED

    np.random.seed(seed)

    # Download and preprocess the data
    # preprocess()

    # Describe each dataset
    # for dataset in DATASETS:
    #     describe(dataset)

    # Mask the keystrokes in each dataset
    # for dataset, param in product(DATASETS, DELAY_PARAMS):
    #     mask_keystrokes(dataset, 'delay', param)
    #
    # for dataset, param in product(DATASETS, INTERVAL_PARAMS):
    #     mask_keystrokes(dataset, 'interval', param)

    # Classify each target and make predictions
    # results = []
    # for mix, dataset in product(MIXES, DATASETS):
    #     unmasked = load_data(dataset)
    #     unmasked_features = extract_features(unmasked)
    #
    #     user_acc, age_acc, gender_acc, hand_acc = (
    #         classification_acc(unmasked_features, target, N_FOLDS) for target in TARGETS
    #     )
    #     pp_smape, dur_smape = predictions_smape(unmasked)
    #
    #     results.append(('none', dataset, 0, 0, user_acc, age_acc, gender_acc, hand_acc, pp_smape, dur_smape))
    #
    #     for param in MIX_PARAMS[mix]:
    #         masked = load_data(dataset, masking=(mix, param))
    #         masked_features = extract_features(masked)
    #         lag = mean_lag(unmasked, masked)
    #
    #         user_acc, age_acc, gender_acc, hand_acc = (
    #             classification_acc(masked_features, target, N_FOLDS) for target in TARGETS
    #         )
    #         pp_smape, dur_smape = predictions_smape(masked)
    #
    #         results.append((mix, dataset, param, lag,
    #                         user_acc, age_acc, gender_acc, hand_acc, pp_smape, dur_smape))
    #
    # results = pd.DataFrame.from_records(results,
    #                                     columns=['strategy', 'dataset', 'param',
    #                                              'mean_delta'] + TARGETS + ['pp_SMAPE', 'dur_SMAPE'])
    #
    # save_results(results, 'results')

    # Make a figure
    acc_figure('results')
