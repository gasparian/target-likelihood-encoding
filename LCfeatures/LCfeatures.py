import numpy as np
import pandas as pd

class LCfeatures:
    
    """Likelihoods / counters features creation.
    Parameters
    https://github.com/gasparian/LC-Features/blob/master/LCfeatures
    ----------
    n_splits : int
        Number of (train_folds+ 1 test_fold) in TimeSeriesSplit
    modes : list od strings
        what kind of features to generate:
            'mean' - mean target value
            'counter' - number of positive class instances
    alpha : float
        regularization coefficient
    features : list of strings / str.: 'all'
        fetures to encode
    target : str.
        target variable name
    """

    def __init__(self, n_splits=9, modes=['mean', 'counter'], alpha=10, features='all', target='conversion'):
        self.alpha = alpha
        self.n_splits = n_splits
        self.modes = modes
        self.features = features
        self.target = target

    def timeSplit(self, df):
        n_samples = len(df)
        n_folds = self.n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds, n_samples))
        indices = df.index.tolist()
        test_size = (n_samples // n_folds)
        test_starts = range(test_size + n_samples % n_folds, n_samples, test_size)
        for test_start in test_starts:
            yield (indices[:test_start], indices[test_start:test_start + test_size])
        
    def fit(self, df):
        if self.features == 'all':
            self.features = list(df.columns).remove(self.target)
        self.values = {
            key: {
                'train': {i:{} for i in range(self.n_splits)},
                'test': {}
                }
            for key in ['mean', 'counter']
        }
        
        fold = 0
        for train_index, test_index in self.timeSplit(df):
            global_mean = df[self.target].loc[train_index].mean()
            for f in self.features:
                groupby_feature = df.loc[train_index].groupby([f])
                current_size = groupby_feature.size()
                if 'counter' in self.modes:
                    self.values['counter']['train'][fold][f] = len(set(df[df[self.target] > 0].index.tolist()).intersection(set(train_index)))

                if 'mean' in self.modes:
                    current_mean = groupby_feature[self.target].mean()
                    self.values['mean']['train'][fold][f] = (current_mean * current_size + global_mean * self.alpha) / (current_size + self.alpha)

            fold += 1
        self.global_mean_test = df[self.target].mean()
        time_test = df.index
        for f in self.features:
            groupby_feature_test = df.groupby([f])
            current_size_test = groupby_feature_test.size()
            if 'counter' in self.modes:
                self.values['counter']['test'][f] = df[df[self.target] > 0].shape[0]

            if 'mean' in self.modes:
                current_mean_test = groupby_feature_test[self.target].mean()
                self.values['mean']['test'][f] = (current_mean_test * current_size_test + self.global_mean_test * self.alpha) / (current_size_test + self.alpha)
    
    def transform(self, df, mode='train'):
        if mode == 'train':
            fold = 0
            for train_index, test_index in self.timeSplit(df):
                globals_ = {
                    "mean":df[self.target].loc[train_index].mean(),
                    "counter":len(set(df[df[self.target] > 0].index.tolist()).intersection(set(train_index)))
                }
                for name in self.modes:
                    for f in self.features:
                        new_name = "%s_%s_LC" % (f, name)
                        if new_name not in df.columns.tolist():
                            df[new_name] = np.nan
                        if fold == 0:
                            df.loc[train_index, new_name] = globals_[name]
                        else:
                            df.loc[test_index, new_name] = self.values[name]['train'][fold][f]
                fold += 1
        elif mode == 'test':
            for name in self.modes:
                for f in self.features:
                    df['%s_%s_LC' % (f, name)] = self.values[name]['test'][f]
        return df