import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from util import keystrokes2events

N_CLUSTERS = 10


def stratified_kfold(df, n_folds):
    """
    Create stratified k-folds from an indexed dataframe
    """
    sessions = pd.DataFrame.from_records(list(df.index.unique())).groupby(0).apply(lambda x: x[1].unique())
    sessions.apply(lambda x: np.random.shuffle(x))
    folds = []
    for i in range(n_folds):
        idx = sessions.apply(lambda x: pd.Series(x[i * (len(x) / n_folds):(i + 1) * (len(x) / n_folds)]))
        idx = pd.DataFrame(idx.stack().reset_index(level=1, drop=True)).set_index(0, append=True).index.values
        folds.append(df.loc[idx])
    return folds


def user_folds(df, target):
    users = df.index.get_level_values(0).unique()
    return [df.loc[u].reset_index().set_index([target, 'session']) for u in users]


class RandomForest():
    def __init__(self, keystroke_feature_fn):
        self.keystroke_feature_fn = keystroke_feature_fn
        self.keystroke_model = RandomForestClassifier(n_estimators=100)

    def fit(self, samples, labels):
        assert len(samples) == len(labels)

        features = []
        for sample in samples:
            features.append(self.keystroke_feature_fn(sample))

        features = pd.concat(features).values
        self.keystroke_model.fit(features, labels)
        return self

    def predict(self, sample):
        features = self.keystroke_feature_fn(sample)
        probas = self.keystroke_model.predict_proba(features)
        scores = dict(zip(self.keystroke_model.classes_, probas.squeeze()))
        max_score_label = max(scores.items(), key=itemgetter(1))[0]
        return max_score_label, scores


def classification_acc(df, target, n_folds):
    if target == 'user':
        folds = stratified_kfold(df, n_folds)
    else:
        folds = user_folds(df, target)

    predictions = []
    for i in range(n_folds):
        print('Fold %d of %d' % (i + 1, n_folds))

        test, train = folds[i], pd.concat(folds[:i] + folds[i + 1:])

        test_labels = test.index.get_level_values(0).values
        train_labels = train.index.get_level_values(0).values

        test_features = pd.concat(test['features'].values).values
        train_features = pd.concat(train['features'].values).values

        cl = RandomForestClassifier(n_estimators=200)
        cl.fit(train_features, train_labels)

        results = cl.predict(test_features)
        predictions.extend(zip([i] * len(test_labels), test_labels, results))

    predictions = pd.DataFrame(predictions, columns=['fold', 'label', 'prediction'])
    summary = predictions.groupby('fold').apply(lambda x: (x['label'] == x['prediction']).sum() / len(x)).describe()

    print('Results')
    print(summary)

    return summary['mean']


def SMAPE(ground_truth, predictions):
    return np.abs((ground_truth - predictions) / (ground_truth + predictions))


def predictions_smape2(df):
    def process_sample(x):
        x = keystrokes2events(x)
        tau = x['time'].diff()
        predictions = pd.expanding_mean(tau).shift()
        return SMAPE(tau, predictions).dropna().mean()

    return df.groupby(level=[0, 1]).apply(process_sample).mean()


def predictions_smape(df):
    def pp_smape(x):
        tau = x['timepress'].diff()
        predictions = pd.expanding_mean(tau).shift()
        return SMAPE(tau, predictions).dropna().mean()

    def duration_smape(x):
        d = x['timerelease'] - x['timepress']
        predictions = pd.expanding_mean(d).shift()
        return SMAPE(d, predictions).dropna().mean()

    return df.groupby(level=[0, 1]).apply(pp_smape).mean(), df.groupby(level=[0, 1]).apply(duration_smape).mean()
