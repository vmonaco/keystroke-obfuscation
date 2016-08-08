import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
FIGURES_DIR = os.path.join(ROOT_DIR, 'figures')
DOWNLOAD_DIR = os.path.join(DATA_DIR, 'download')


def keystrokes2events(df):
    time_cols = [c for c in df.columns if 'time' in c]
    dfs = []
    for col in time_cols:
        new_df = df[df.columns.difference(time_cols)]
        new_df['time'] = df[col]
        new_df['action'] = col
        dfs.append(new_df)

    df = pd.concat(dfs).sort_values('time')

    return df


def events2keystrokes(df):
    dfs = []
    keys_down = {}
    for idx, row in df.iterrows():
        if row['action'] == 'timepress':
            if row['keyname'] in keys_down.keys():
                raise Exception('Warning: key pressed twice without release (probably auto repeated while held down)')
            else:
                keys_down[row['keyname']] = row
        elif row['action'] == 'timerelease':
            new_row = row
            if new_row['keyname'] not in keys_down:
                raise Exception('Warning: key released without first being pressed')
            else:
                new_row['timepress'] = keys_down[new_row['keyname']]['time']

            new_row['timerelease'] = new_row['time']
            del new_row['time']
            del new_row['action']
            del keys_down[row['keyname']]
            dfs.append(new_row)

    df = pd.concat(dfs, axis=1).T.sort_values('timepress')
    df.index.names = ['user', 'session']

    return df


def reduce_dataset(df, num_users=None,
                   min_samples=None, max_samples=None,
                   min_obs=None, max_obs=None):
    """
    Reducing the size of a dateset is a common operation when a certain number
    of observations, samples, or users is desired. This function limits each
    of these by attempting to satisfy the constraints in the following order:

        num observations
        num samples
        num users

    """

    if max_obs:
        df = df.groupby(level=[0, 1]).apply(lambda x: x[:max_obs]).reset_index(level=[2, 3], drop=True)

    num_obs = df.groupby(level=[0, 1]).size()

    if min_obs:
        num_obs = num_obs[num_obs >= min_obs]

    num_samples = num_obs.groupby(level=0).size()

    if min_samples:
        num_samples = num_samples[num_samples >= min_samples]

    if num_users and num_users < len(num_samples):
        users = np.random.permutation(num_samples.index.values)[:num_users]
    else:
        users = num_samples.index.values

    num_obs = num_obs.loc[users.tolist()]

    if max_samples:
        num_obs = num_obs.groupby(level=0).apply(
                lambda x: x.loc[np.random.permutation(np.sort(x.index.unique()))[:max_samples]]).reset_index(level=1,
                                                                                                             drop=True)

    df = df.loc[num_obs.index].sort_index()

    return df


def load_data(name, masking=None, **args):
    df = pd.read_csv(os.path.join(DATA_DIR, name + '.csv' if masking is None else name + '_%s%s.csv' % masking),
                     index_col=[0, 1], **args)

    return df


def save_data(df, name, masking=None, **args):
    df.to_csv(os.path.join(DATA_DIR, name + '.csv' if masking is None else name + '_%s%s.csv' % masking), **args)
    return


def save_fig(name, ext='pdf'):
    plt.savefig(os.path.join(FIGURES_DIR, name + '.%s' % ext), bbox_inches='tight')
    plt.close()
    return


def save_results(df, name):
    df.to_csv(os.path.join(RESULTS_DIR, name + '.csv'))
    return


def load_results(name):
    return pd.read_csv(os.path.join(RESULTS_DIR, name + '.csv'), index_col=0)
