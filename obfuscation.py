import numpy as np
from util import load_data, save_data, keystrokes2events, events2keystrokes


def delay_mix(df, Delta):
    time = df['time'].values
    new_time = np.zeros(len(time))
    delays = np.zeros(len(time))
    delays[0] = np.random.uniform(0, Delta)
    new_time[0] = time[0] + delays[0]
    lb = np.zeros(len(time))
    for i in range(1, len(time)):
        lb[i] = max(new_time[i - 1] - time[i], 0)
        delays[i] = np.random.uniform(lb[i], Delta)
        new_time[i] = time[i] + delays[i]

    df['time'] = new_time
    return df


def interval_mix(df, b=1, u0=1, eps=1):
    actual_time = df['time'].values
    jammed_time = np.zeros(len(df))
    jammed_time[0] = df['time'].values[0]

    u = u0
    for i in range(1, len(df)):
        desired_tau = np.random.uniform(0, u)
        desired_time = jammed_time[i - 1] + desired_tau
        jammed_time[i] = max(desired_time, actual_time[i])
        u = u + b * (actual_time[i] - desired_time)
        u = max(u, eps)

    df['time'] = jammed_time
    return df


def obfuscate_keystrokes(name, strategy, param):
    """

    """
    df = load_data(name)
    df = df.groupby(level=[0, 1]).apply(keystrokes2events).reset_index(level=[2, 3], drop=True)

    if strategy == 'delay':
        df = df.groupby(level=[0, 1]).apply(lambda x: delay_mix(x, param))
    elif strategy == 'interval':
        df = df.groupby(level=[0, 1]).apply(lambda x: interval_mix(x, param))
    else:
        raise Exception('Unknown masking strategy')

    df = df.groupby(level=[0, 1]).apply(events2keystrokes).reset_index(level=[2, 3], drop=True)
    save_data(df, name, masking=(strategy, param))
    return


def mean_lag(unmasked, masked):
    unmasked = keystrokes2events(unmasked)
    masked = keystrokes2events(masked)
    return (masked['time'] - unmasked['time']).mean()
