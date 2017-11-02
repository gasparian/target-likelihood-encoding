import numpy as np
import pandas as pd

class LCfeatures(object):
    
    """Likelihoods / counters features creation.
    Parameters
    ----------

    n_splits : int
        Number of (train_folds+ 1 test_fold) in TimeSeriesSplit
    modes : list od strings
        what kind of features to generate:
            'std' - standart deviation of target
            'mean' - mean target value
            'counter' - number class instances
    alpha : float
        regularization coefficient
    features : list of strings / str.: 'all'
        which fwatures to encode
    target : str.
        name of target variable

    Returns
    -------
    likelihood / counters sets: dict of DataFrames
    transformed DataFrame : Pandas DataFrame

    """

    def __init__(self, nfolds=10, modes=['mean', 'std', 'counter'], alpha=10, features='all', target='conversion'):
        self.alpha = alpha
        self.nfolds = nfolds
        self.modes = modes
        self.features = features
        self.target = target

    def timeSplit(self, df, folds=5):
        idx = list(df.index.values)
        folds += 1
        length = len(idx)
        fold_size = length // folds
        remain = length - folds*fold_size
        steps = list(range(0, length, fold_size))
        train_index, test_index = [], []
        for i, step in enumerate(steps[:-2]):
            train_index.append(idx[:steps[i+1]+remain])
            test_index.append(idx[steps[i+1]+remain:steps[i+2]+remain])
        return zip(train_index, test_index)
        
    def fit(self, df):
        try:
            df.drop([i for i in df.columns if re.search('_SA', i)], axis=1, inplace=True)
        except:
            pass
        if self.features == 'all':
            self.features = list(df.columns).remove(self.target)
        self.values = {
            'mean': {
                'train': {i:{} for i in range(1,self.nfolds+1)},
                'test': {}
            },
            'counts': {
                'train': {i:{} for i in range(1,self.nfolds+1)},
                'test': {}
            },
            'stds': {
                'train': {i:{} for i in range(1,self.nfolds+1)},
                'test': {}
            }
        }

        fold = 1
        for train_index, test_index in self.timeSplit(df, folds=self.nfolds):
            if 'mean' in self.modes:
                global_mean = df[self.target].loc[train_index].mean()
            if 'std' in self.modes:
                global_std = df[self.target].loc[train_index].std()
            for f in self.features:
                groupby_feature = df.loc[train_index].groupby([f])
                current_size = groupby_feature.size()
                if 'counts' in self.modes:
                    self.values['counts']['train'][fold][f] = pd.DataFrame(current_size, columns=["%s_counts_SA" % f], index=current_size.index, dtype=np.float64)

                if 'mean' in self.modes:
                    current_mean = groupby_feature[self.target].mean()
                    feat_df = ((current_mean * current_size + global_mean * self.alpha) / (current_size + self.alpha)).fillna(global_mean)
                    self.values['mean']['train'][fold][f] = pd.DataFrame(feat_df, columns=["%s_mean_SA" % f], index=feat_df.index, dtype=np.float64)

                if 'std' in self.modes:
                    current_std = groupby_feature[self.target].std()
                    feat_df = ((current_std * current_size + global_std * self.alpha) / (current_size + self.alpha)).fillna(global_std)
                    self.values['stds']['train'][fold][f] = pd.DataFrame(feat_df, columns=["%s_stds_SA" % f], index=feat_df.index, dtype=np.float64)
            fold += 1
        if 'mean' in self.modes:
            self.global_mean_test = df[self.target].mean()
        if 'std' in self.modes:
            self.global_std_test = df[self.target].std()
        time_test = df.index
        for f in self.features:
            groupby_feature_test = df.groupby([f])
            current_size_test = groupby_feature_test.size()
            if 'counts' in self.modes:
                self.values['counts']['test'][f] = pd.DataFrame(current_size_test, columns=["%s_counts_SA" % f], index=current_size_test.index, dtype=np.float64)

            if 'mean' in self.modes:
                current_mean_test = groupby_feature_test[self.target].mean()
                feat_df_test = ((current_mean_test * current_size_test + self.global_mean_test * self.alpha) / (current_size_test + self.alpha)).fillna(self.global_mean_test)
                self.values['mean']['test'][f] = pd.DataFrame(feat_df_test, columns=["%s_mean_SA" % f], index=feat_df_test.index, dtype=np.float64)

            if 'std' in self.modes:
                current_std_test = groupby_feature_test[self.target].std()
                feat_df_test = ((current_std_test * current_size_test + self.global_std_test * self.alpha) / (current_size_test + self.alpha)).fillna(self.global_std_test)
                self.values['stds']['test'][f] = pd.DataFrame(feat_df_test, columns=["%s_stds_SA" % f], index=feat_df_test.index, dtype=np.float64)

    
    def transform(self, df, mode='train'):
        try:
            df.drop([i for i in df.columns if re.search('_SA', i)], axis=1, inplace=True)
        except:
            pass
        for name in self.modes:
            lvl2 = None
            if mode == 'train':
                fold = 1
                for train_index, test_index in self.timeSplit(df, folds=self.nfolds):
                    if name == 'mean':
                        global_ = df[self.target].loc[train_index].mean()
                    if name == 'std':
                        global_ = df[self.target].loc[train_index].std()
                    lvl1 = None
                    for f in self.features:
                        if lvl1 is not None:
                            lvl1 = pd.merge(lvl1, self.values[name]['train'][fold][f], how="left", left_on=f, right_index=True)
                        else:
                            if fold == 1:
                                lvl1 = pd.merge(df.loc[train_index], pd.DataFrame({f: global_}, index=range(1)), how="left", left_on=f, right_index=True)
                                lvl1_test = pd.merge(df.loc[test_index], self.values[name]['train'][fold][f], how="left", left_on=f, right_index=True)
                                lvl1 = pd.concat([lvl1, lvl1_test], axis=0, ignore_index=False, join='outer', copy=False)
                                del lvl1_test
                            else:
                                lvl1 = pd.merge(df.loc[test_index], self.values[name]['train'][fold][f], how="left", left_on=f, right_index=True)
                        if name != 'counts':
                            lvl1['%s_%s_SA' % (f, name)] = lvl1['%s_%s_SA' % (f, name)].fillna(global_)
                    if lvl2 is not None:
                        lvl2 = pd.concat([lvl2, lvl1], axis=0, ignore_index=False, join='outer', copy=False)
                    else:
                        lvl2 = lvl1
                    fold += 1
                df = lvl2
                del lvl1
                del lvl2
            elif mode == 'test':
                for name in self.modes:
                    for f in self.features:
                        df = pd.merge(df, self.values[name]['test'][f], how="left", left_on=f, right_index=True)
                        if name != 'counts':
                            df['%s_%s_SA' % (f, name)] = df['%s_%s_SA' % (f, name)].fillna(self.global_mean_test)
        return df