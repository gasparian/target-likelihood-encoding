import numpy as np
import pandas as pd

class LCfeatures:
    
    """Likelihoods features creation.
    Parameters
    https://github.com/gasparian/LC-Features/blob/master/LCfeatures
    ----------
    n_splits : int
        Number of (train_folds+ 1 test_fold) in TimeSeriesSplit
    modes : list od strings
        what kind of features to generate: 'mean', 'std', 'min', 'max'
    alpha : float
        regularization coefficient
    features : list of strings / str.: 'all'
        fetures to encode
    target : str.
        target variable name
    """

    def __init__(self, n_splits=9, modes=['mean'], alpha=10, features='all', target='conversion'):
        self.alpha = alpha
        self.n_splits = n_splits
        self.modes = modes # 'std', 'mean', 'max', 'min'
        self.features = features
        self.target = target
        self.calculator = {
            "mean": lambda : self.smoothed_encoding(groupby_feature[self.target].mean(), current_size, global_mean),
            "std": lambda : self.smoothed_encoding(groupby_feature[self.target].std(), current_size, global_std),
            "min": lambda : self.smoothed_encoding(groupby_feature[self.target].min(), current_size, global_min),
            "max": lambda : self.smoothed_encoding(groupby_feature[self.target].max(), current_size, global_max)
        }

    def smoothed_encoding(self, current, current_size, global_val):
        return (current * current_size + global_val * self.alpha) / (current_size + self.alpha)

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
            for key in self.modes
        }
        
        fold = 0
        for train_index, test_index in self.timeSplit(df):
            global_mean = df[self.target].loc[train_index].mean()
            global_std = df[self.target].loc[train_index].std()
            global_min = df[self.target].loc[train_index].min()
            global_max = df[self.target].loc[train_index].max()
            for f in self.features:
                groupby_feature = df.loc[train_index].groupby([f])
                current_size = groupby_feature.size()
                for mode in self.modes:
                    self.values[mode]['train'][fold][f] = self.calculator[mode]()

            fold += 1

        global_mean = df[self.target].mean()
        global_std = df[self.target].std()
        global_min = df[self.target].min()
        global_max = df[self.target].max()
        for f in self.features:
            groupby_feature = df.groupby([f])
            current_size = groupby_feature_test.size()
            for mode in self.modes:
                self.values[mode]['test'][f] = self.calculator[mode]()
    
    def transform(self, df, mode='train'):
        if mode == 'train':
            fold = 0
            for train_index, test_index in self.timeSplit(df):
                global_vals = {
                    "mean":df[self.target].loc[train_index].mean() if "mean" in self.modes else None,
                    "std": df[self.target].loc[train_index].std() if "std" in self.modes else None,
                    "max":df[self.target].loc[train_index].max() if "max" in self.modes else None,
                    "min": df[self.target].loc[train_index].min() if "min" in self.modes else None,
                }
                for name in self.modes:
                    for f in self.features:
                        new_name = "%s_%s_LC" % (f, name)
                        if new_name not in df.columns.tolist():
                            df[new_name] = np.nan
                        if fold == 0:
                            df.loc[train_index, new_name] = global_vals[name]
                        else:
                            df.loc[test_index, new_name] = self.values[name]['train'][fold][f]
                fold += 1
        elif mode == 'test':
            for name in self.modes:
                for f in self.features:
                    df['%s_%s_LC' % (f, name)] = self.values[name]['test'][f]
        return df