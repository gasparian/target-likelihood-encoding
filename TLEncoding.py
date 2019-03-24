import numpy as np
import pandas as pd

class TLEncoding:
    
    """
    Target likelihood features.
    Parameters
    https://github.com/gasparian/target-likelihood-encoding/blob/master/TLEncoding
    ----------
    n_splits : int
        Number of (train_folds+ 1 test_fold) in TimeSeriesSplit
    alpha : float
        regularization coefficient
    target : str.
        target variable name
    """

    def __init__(self, n_splits=9, alpha=10, target='conversion'):
        
        self.alpha = alpha
        self.n_splits = n_splits
        self.modes = ["mean", "std"]
        self.target = target
        self.calculator = {
            "mean": lambda groupby_feature, current_size, global_mean: self.smoothed_encoding(groupby_feature[self.target].mean(), current_size, global_mean),
            "std": lambda groupby_feature, current_size, global_std: self.smoothed_encoding(groupby_feature[self.target].std(), current_size, global_std)
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
            
    def get_globs(self, df):
        return {
            "mean":df.mean(),
            "std":df.std()
        }
        
    def fit(self, df):
        
        self.features = list(df.columns)
        self.features.remove(self.target)

        self.values = {
            key: {
                'train': {i:{} for i in range(self.n_splits)},
                'test': {}
                }
            for key in self.modes
        }
        
        fold = 0
        for train_index, test_index in self.timeSplit(df):
            globs = self.get_globs(df.loc[train_index, self.target])
            for f in self.features:
                groupby_feature = df.loc[train_index].groupby([f])
                current_size = groupby_feature.size()
                for mode in self.modes:
                    self.values[mode]['train'][fold][f] = \
                        self.calculator[mode](groupby_feature, current_size, globs[mode]).reset_index(drop=False).fillna(0)
            fold += 1
            
        globs = self.get_globs(df[self.target])
        for f in self.features:
            groupby_feature = df.groupby([f])
            current_size = groupby_feature.size()
            for mode in self.modes:
                self.values[mode]['test'][f] = \
                    self.calculator[mode](groupby_feature, current_size, globs[mode]).reset_index(drop=False).fillna(0)
    
    def transform(self, df, mode='train'):
        new_df = pd.DataFrame(index=df.index)
        if mode == 'train':
            # target must be included in train mode
            fold = 0
            for train_index, test_index in self.timeSplit(new_df):
                globs = self.get_globs(df.loc[train_index, self.target])
                for name in self.modes:
                    # using the same features as for fitting
                    for f in self.features:
                        new_name = "%s_%s_TL" % (f, name)
                            
                        if fold == 0:
                            new_df.loc[train_index, new_name] = globs[name]
                            
                        new_df.loc[test_index, new_name] = \
                            pd.merge(df.loc[test_index, [f]], self.values[name]['train'][fold][f], 
                                     how="left", on=f)[0].values
                fold += 1
                
        elif mode == 'test':
            for name in self.modes:
                for f in self.features:
                    new_df['%s_%s_TL' % (f, name)] = \
                        pd.merge(df[[f]], self.values[name]['test'][f], 
                                 how="left", on=f)[0].values
        return new_df