import numpy as np
import pandas as pd
import keycode

FALLBACK_WEIGHT = 0.25  # weight of fallback observations
M_MIN_FREQUENCY = 3  # min frequency per sample for feature fallback

OUTLIER_DISTANCE = 2  # outliers outside +/- std devs
OUTLIER_ITERATIONS = 2  # no. iterations to do recursive outlier removal


def transition_digrams(df, distance=1):
    a = df.groupby(['user', 'session']).apply(lambda x: x[:-distance].reset_index())
    b = df.groupby(['user', 'session']).apply(lambda x: x[distance:].reset_index())

    a = a[['user', 'session', 'keyname', 'timepress', 'timerelease']]
    b = b[['keyname', 'timepress', 'timerelease']]

    a.columns = ['user', 'session', 'keyname_1', 'timepress_1', 'timerelease_1']
    b.columns = ['keyname_2', 'timepress_2', 'timerelease_2']

    joined = pd.concat([a, b], join='inner', axis=1)

    cols = ['user', 'session', 'keynames', 'transition']

    # Create columns for each transition type
    t1 = pd.DataFrame({'user': joined['user'],
                       'session': joined['session'],
                       'keynames': joined['keyname_1'] + '__' + joined['keyname_2'],
                       'transition': joined['timepress_2'] - joined['timerelease_1']},
                      columns=cols, index=joined.index)

    t2 = pd.DataFrame({'user': joined['user'],
                       'session': joined['session'],
                       'keynames': joined['keyname_1'] + '__' + joined['keyname_2'],
                       'transition': joined['timepress_2'] - joined['timepress_1']},
                      columns=cols, index=joined.index)

    return t1, t2


def outlier_removal_recursive(df, col, std_distance=OUTLIER_DISTANCE, max_iterations=OUTLIER_ITERATIONS):
    '''
    Remove duration outliers on a per-user basis

    10 iterations will remove most outliers.

    Does the following:
        group df by user and keyname
        get mean and std for each group (user/keyname combination)
        filter df durations with the corresponding user/key mean and stds

    This could be more efficient by testing the number of outliers removed for
    each group and only recomputing the groups with more than 0 removed
    '''
    prev_len = np.inf
    i = 0
    while prev_len > len(df):
        prev_len = len(df)
        df = outlier_removal(df, col, std_distance=std_distance)
        print('Removed %d observations' % (prev_len - len(df)))

        i += 1
        if max_iterations > 0 and i == max_iterations:
            break

    return df


def outlier_removal(df, col, std_distance=4):
    '''
    Remove duration outliers on a per-user basis

    10 iterations will remove most outliers.

    Does the following:
        group df by user and keyname
        get mean and std for each group (user/keyname combination)
        filter df durations with the corresponding user/key mean and stds

    This could be more efficient by testing the number of outliers removed for
    each group and only recomputing the groups with more than 0 removed
    '''

    m, s = df[col].mean(), df[col].std()

    lower = m - std_distance * s
    upper = m + std_distance * s

    df = df[(df[col].values > lower) & (df[col].values < upper)]
    return df


def reverse_tree(features, hierarchy):
    parents = {}

    for parent, children in hierarchy.items():
        for child in children:
            parents[child] = parent

    return parents


def extract_gaussian_features(df, group_col_name, feature_col_name, features, decisions, feature_name_prefix):
    feature_vector = {}
    for feature_name, feature_set in features.items():
        full_feature_name = '%s%s' % (feature_name_prefix, feature_name)

        obs = df.loc[df[group_col_name].isin(feature_set), feature_col_name]

        if len(obs) < M_MIN_FREQUENCY and feature_name in decisions.keys():
            fallback_name = decisions[feature_name]
            fallback_obs = pd.DataFrame()
            while len(obs) + len(fallback_obs) < M_MIN_FREQUENCY:
                fallback_set = getattr(keycode, fallback_name)
                fallback_obs = df.loc[df[group_col_name].isin(fallback_set), feature_col_name]

                if fallback_name in decisions.keys():
                    fallback_name = decisions[fallback_name]  # go up the tree
                else:
                    break  # reached the root node

            n = len(obs)

            # Prevent NA values
            if n == 0:
                obs_mean = 0
                obs_std = 0
            elif n == 1:
                obs_mean = obs.mean()
                obs_std = 0
            else:
                obs_mean = obs.mean()
                obs_std = obs.std()

            feature_vector['%s.mean' % full_feature_name] = (n * obs_mean + FALLBACK_WEIGHT * fallback_obs.mean()) / (
                n + FALLBACK_WEIGHT)
            feature_vector['%s.std' % full_feature_name] = (n * obs_std + FALLBACK_WEIGHT * fallback_obs.std()) / (
                n + FALLBACK_WEIGHT)
        else:
            feature_vector['%s.mean' % full_feature_name] = obs.mean()
            feature_vector['%s.std' % full_feature_name] = obs.std()

    return pd.Series(feature_vector)


def keystroke_durations(df):
    return pd.DataFrame(
            {'keyname': df['keyname'].values, 'duration': df['timerelease'].values - df['timepress'].values})


def keystroke_transitions(df):
    keynames = df[:-1]['keyname'].values + '__' + df[1:]['keyname'].values
    t1 = df[1:]['timepress'].values - df[:-1]['timerelease'].values
    t2 = df['timepress'].diff().dropna().values
    t3 = df['timerelease'].diff().dropna().values
    t4 = df[1:]['timerelease'].values - df[:-1]['timepress'].values

    return pd.DataFrame({'keynames': keynames, 't1': t1, 't2': t2, 't3': t3, 't4': t4})


def clean_features(df):
    df[(df == np.inf) | (df == -np.inf) | (np.isnan(df))] = 0
    return df


def durations_transitions(df):
    df = df.groupby(level=[0, 1]).apply(lambda x: x.reset_index().sort('timepress')).reset_index(level=2, drop=True)
    d = df.groupby(level=[0, 1]).apply(keystroke_durations).reset_index(level=2, drop=True)
    t = df.groupby(level=[0, 1]).apply(keystroke_transitions).reset_index(level=2, drop=True)
    return d, t


def extract_keystroke_features(df):
    d, t = durations_transitions(df)

    features = keycode.LINGUISTIC_FEATURES
    fallback = keycode.LINGUISTIC_FALLBACK

    duration_features = {k: v for k, v in features.items() if '__' not in k}
    transition_features = {k: v for k, v in features.items() if '__' in k}
    decisions = reverse_tree(features, fallback)

    if len(duration_features) > 0:
        du = outlier_removal_recursive(d, 'duration')

        du_features = du.groupby(level=[0, 1]).apply(lambda x:
                                                     extract_gaussian_features(x, feature_col_name='duration',
                                                                               group_col_name='keyname',
                                                                               features=duration_features,
                                                                               decisions=decisions,
                                                                               feature_name_prefix='du_'))

    if len(transition_features) > 0:
        t1 = outlier_removal_recursive(t[['keynames', 't1']], 't1')
        t2 = outlier_removal_recursive(t[['keynames', 't2']], 't2')

        t1_features = t1.groupby(level=[0, 1]).apply(lambda x:
                                                     extract_gaussian_features(x, feature_col_name='t1',
                                                                               group_col_name='keynames',
                                                                               features=transition_features,
                                                                               decisions=decisions,
                                                                               feature_name_prefix='t1_'))
        t2_features = t2.groupby(level=[0, 1]).apply(lambda x:
                                                     extract_gaussian_features(x, feature_col_name='t2',
                                                                               group_col_name='keynames',
                                                                               features=transition_features,
                                                                               decisions=decisions,
                                                                               feature_name_prefix='t2_'))

    fspace = pd.concat([du_features, t1_features, t2_features], axis=1)
    fspace = clean_features(fspace)

    return fspace
