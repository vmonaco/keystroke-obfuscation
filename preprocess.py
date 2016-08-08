import os
import numpy as np
import pandas as pd

from util import DATA_DIR, DOWNLOAD_DIR, load_data, reduce_dataset, keystrokes2events, events2keystrokes

GREYC_NISLAB_DATASET_URL = 'http://www.epaymentbiometrics.ensicaen.fr/wp-content/uploads/2015/04/greyc-nislab-keystroke-benchmark-dataset.xls'
CITEFA_DATASET_URL = 'http://www.cse.chalmers.se/~bello/publications/kprofiler-20100716-1442.tar.gz'
VILLANI_DATASET_URL = 'https://bitbucket.org/vmonaco/dataset-villani-keystroke/raw/f451aa1b1ee40e86ef58d8eab2b8f829fcc23405/data/keystroke.csv'

COLS = ['age', 'gender', 'handedness', 'timepress', 'timerelease', 'keyname']


def make_sessions(x, mean, std, skip_chars=20):
    x = x.reset_index(level=1)
    base_session = x['session'].unique().squeeze() * 10000
    end = 0
    session = base_session
    while end < len(x):
        if std > 0:
            new_end = end + int(np.random.normal(mean, std))
        else:
            new_end = end + int(mean)

        if new_end > len(x):
            x['session'][end:new_end] = -1
            break
        else:
            x['session'][end:new_end] = session

        x['session'][new_end:new_end + skip_chars] = -1
        session += 1
        end = new_end + skip_chars

    x = x[x['session'] >= 0]
    x = x.set_index('session', append=True)

    return x


def preprocess_greyc_nislab(in_file, out_file):
    """
    Preprocess the raw GREYC NISLAB dataset
    """
    df = pd.concat([pd.read_excel(in_file, sheetname=0),
                    pd.read_excel(in_file, sheetname=1),
                    pd.read_excel(in_file, sheetname=2),
                    pd.read_excel(in_file, sheetname=3),
                    pd.read_excel(in_file, sheetname=4)])

    df = df[df['Class'] == 2]

    df['age'] = (df['Age'] < 30).map({True: '<30', False: '>=30'})
    df['gender'] = df['Gender'].map({'F': 'female', 'M': 'male'})
    df['handedness'] = df['Handedness'].map({'L': 'left', 'R': 'right'})
    df['session'] = np.arange(len(df))

    df['password'] = df['Password'].map({
        'leonardo dicaprio': 1,
        'the rolling stones': 2,
        'michael schumacher': 3,
        'red hot chilli peppers': 4,
        'united states of america': 5,
    })

    def preprocess_row(idx_row):
        idx, row = idx_row
        keyname = list(map(lambda x: 'space' if x == ' ' else x, list(row['Password'])))
        v = np.array(row['Keystroke Template Vector'].strip().split()).astype(int) // 10000

        s = len(keyname) - 1
        pp, rr, pr, rp = [v[s * i:s * (i + 1)] for i in range(4)]

        timepress = np.r_[0, pp].cumsum()

        # Offset the first release time by the duration of the first key
        timerelease = np.r_[rp[0] - rr[0], rr].cumsum()

        # There are ~180 rows where timerelease == timepress.
        # Fix these by assuming at least the minimum standard clock resolution
        timerelease[timerelease == timepress] += 16
        sample = pd.DataFrame.from_items([
            ('user', row['User_ID']),
            ('session', row['session']),
            ('password', row['password']),
            ('age', row['age']),
            ('gender', row['gender']),
            ('handedness', row['handedness']),
            ('timepress', timepress),
            ('timerelease', timerelease),
            ('keyname', keyname)
        ])

        return sample

    df = pd.concat(map(preprocess_row, df.iterrows()))
    df = df.set_index(['user', 'session'])[COLS]
    df = remove_repeated_keys(df)
    df.to_csv(out_file)
    return


def preprocess_citefa(in_file, out_file):
    """
    Preprocess the raw CITEFA dataset
    """
    import tempfile
    import shutil
    import tarfile
    from glob import glob
    from operator import itemgetter

    from keycode import lookup_key, detect_agent

    tdir = tempfile.mkdtemp()
    tfile = tarfile.open(in_file, 'r:gz')
    tfile.extractall(tdir)

    dfs = []
    for fname in glob(os.path.join(tdir, '*', '*')):
        with open(fname) as f:
            lines = f.readlines()

        header = lines[0]
        agent = detect_agent(header)
        fields = header.split(';')

        age = '<30' if int(fields[7]) < 30 else '>=30'
        gender = 'male' if fields[8] == 'Male' else 'female'
        handedness = 'right' if fields[9] == 'right-handed' else 'left'

        # rows contain the keypress/keyrelease actions, need to convert to keystrokes
        key_actions = [row.strip().split() for row in lines if ('dn' in row) or ('up' in row)]

        # parse the ints
        key_actions = [(i1, int(i2), i3, int(i4)) for i1, i2, i3, i4 in key_actions]
        key_actions = sorted(key_actions, key=itemgetter(1))

        keystrokes = []
        keys_down = {}

        for task, time, action, keycode in key_actions:
            if action == 'dn':
                if keycode in keys_down.keys():
                    print('Warning: key pressed twice without release (probably auto repeated while held down)')
                    continue
                keys_down[keycode] = time
            elif action == 'up':
                if keycode not in keys_down.keys():
                    print('Warning: key released without first being pressed', time, keycode)
                    continue
                keystrokes.append((task, keys_down[keycode], time, lookup_key(keycode, agent)))
                del keys_down[keycode]
            else:
                raise Exception('Unknown action')

        task, timepress, timerelease, keyname = zip(*keystrokes)

        dfs.append(pd.DataFrame.from_items([
            ('user', fields[4]),
            ('session', int(fields[2])),
            ('age', age),
            ('gender', gender),
            ('handedness', handedness),
            ('task', task),
            ('timepress', timepress),
            ('timerelease', timerelease),
            ('keyname', keyname)
        ]))
    shutil.rmtree(tdir)

    df = pd.concat(dfs)

    # Keep only the sentence copy tasks. See Bello 2010
    df = df[df['task'].isin(
            {'ks_00', 'ks_01', 'ks_02', 'ks_03', 'ks_04', 'ks_05',
             'ks_06', 'ks_07', 'ks_08', 'ks_09', 'ks_10',
             'ks_11' 'ks_12', 'ks_13', 'ks_14'})]

    df['session'] = df['session'] * 100 + df['task'].str[3:].astype(int)

    df = df.set_index(['user', 'session'])
    df = remove_repeated_keys(df)
    df = reduce_dataset(df, min_samples=10, max_samples=10)
    df.to_csv(out_file)
    return


def preprocess_villani(in_file, out_file, long_fixed_out_file):
    """
    Preprocess the raw Villani dataset and extend the long fixed dataset
    """
    df = pd.read_csv(in_file, index_col=[0, 1])

    # Make age a binary target, <30 and >=30
    df['age'] = df['agegroup'].map({
        'under20': '<30',
        '20-29': '<30',
        '30-39': '>=30',
        '40-49': '>=30',
        '50-59': '>=30',
        'over60': '>=30'}
    )

    # Ignore missing data
    df = df.dropna()
    df = remove_repeated_keys(df)

    # combine the villani fixed text with citefa dataset fixed text
    long_fixed = load_data('long_fixed')
    slf = long_fixed.groupby(level=[0, 1]).size()

    villani_fixed = df[df['inputtype'] == 'fixed']
    villani_fixed = villani_fixed.groupby(level=[0, 1]).apply(lambda x: make_sessions(x, slf.mean(), slf.std()))
    villani_fixed = villani_fixed.reset_index(level=[0, 1], drop=True)
    villani_fixed = reduce_dataset(villani_fixed, min_samples=10, max_samples=10)

    long_fixed = pd.concat([long_fixed, villani_fixed])
    long_fixed = long_fixed[COLS]
    long_fixed.to_csv(long_fixed_out_file)

    # Free-text input only
    villani_free = df[df['inputtype'] == 'free']
    villani_free = villani_free.groupby(level=[0, 1]).apply(lambda x: make_sessions(x, slf.mean(), slf.std()))
    villani_free = villani_free.reset_index(level=[0, 1], drop=True)

    villani_free = reduce_dataset(villani_free, min_samples=10, max_samples=10)
    villani_free = villani_free[COLS]
    villani_free.to_csv(out_file)
    return


def remove_repeated_keys(df):
    def process_sample(x):
        dfs = []
        last_release = {}
        for idx, row in x.iterrows():
            # time press must be after last release, otherwise ignore
            if row['keyname'] in last_release.keys() and row['timepress'] <= last_release[row['keyname']]:
                continue

            last_release[row['keyname']] = row['timerelease']
            dfs.append(row)

        x = pd.concat(dfs, axis=1).T
        x.index.names = ['user', 'session']
        return x

    df = df.groupby(level=[0, 1]).apply(process_sample).reset_index(level=[2, 3], drop=True)
    return df


def preprocess():
    """
    Download and preprocess datasets for the experiments.
    """
    import urllib.request
    import urllib.error

    def download_dataset(name, local_name, url):
        if os.path.exists(os.path.join(DOWNLOAD_DIR, local_name)):
            print('Already downloaded %s' % name)
            return

        try:
            print('Downloading %s' % name)
            urllib.request.urlretrieve(url, os.path.join(DOWNLOAD_DIR, local_name))
        except urllib.error.HTTPError as e:
            print('WARNING: Unable to download %s from URL:\n%s' % (name, url))
            print('Check that the URL is correct and you have permissions to download the file.')

    # Download both datasets
    download_dataset('GREYC NISLAB Dataset', 'greyc_nislab.xls', GREYC_NISLAB_DATASET_URL)
    download_dataset('CITAFA Dataset', 'citefa.tar.gz', CITEFA_DATASET_URL)
    download_dataset('Villani Dataset', 'villani.csv', VILLANI_DATASET_URL)

    # This creates the short fixed dataset
    # preprocess_greyc_nislab(os.path.join(DOWNLOAD_DIR, 'greyc_nislab.xls'),
    #                         os.path.join(DATA_DIR, 'short_fixed.csv'))

    # This creates the long fixed dataset
    preprocess_citefa(os.path.join(DOWNLOAD_DIR, 'citefa.tar.gz'),
                      os.path.join(DATA_DIR, 'long_fixed.csv'))

    # This creates the long free dataset and extends the previous long fixed dataset
    preprocess_villani(os.path.join(DOWNLOAD_DIR, 'villani.csv'),
                       os.path.join(DATA_DIR, 'long_free.csv'),
                       os.path.join(DATA_DIR, 'long_fixed.csv'))

    return
