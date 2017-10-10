class LCfeatures(object):
    
    """Likelihoods / counters features creation.
    Parameters
    ----------

    cv : cross validation strategy instance
        Ex.: KFold(n_splits=5)
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

    def __init__(self, cv, modes=['mean', 'std', 'counter'], alpha=10, features='all', target='conversion'):
        self.path = path
        self.alpha = alpha
        self.modes = modes
        self.cv = cv
        self.features = features
        self.target = target
        
    def fit(self, df):
        try:
            df.drop([i for i in df.columns if re.search('_SA', i)], axis=1, inplace=True)
        except:
            pass
        if features == 'all':
            self.features = list(df.columns).remove(self.target)
        self.nfolds = self.cv.get_n_splits()
        self.mean_train = {i:None for i in range(1,self.nfolds+1)}
        self.mean_test = {}
        self.counts_train = {i:None for i in range(1,self.nfolds+1)}
        self.counts_test = {}
        self.stds_train = {i:None for i in range(1,self.nfolds+1)}
        self.stds_test = {}
        fold = 1
        for train_index, test_index in self.cv.split(df):
            if 'mean' in self.modes:
                global_mean = df[self.target].loc[train_index].mean()
            if 'std' in self.modes:
                global_std = df[self.target].loc[train_index].std()
            for f in self.features:
                groupby_feature = df.loc[train_index].groupby([f])
                current_size = groupby_feature.size()
                if 'counts' in self.modes:
                    self.counts_train[fold][f] = pd.DataFrame(current_size, columns=["%s_counts_SA" % f], index=current_size.index, dtype=np.float64)

                if 'mean' in self.modes:
                    current_mean = groupby_feature[self.target].loc[train_index].mean()
                    feat_df = ((current_mean * current_size + global_mean * self.alpha) / (current_size + self.alpha)).fillna(global_mean)
                    self.mean_train[fold][f] = pd.DataFrame(feat_df, columns=["%s_mean_SA" % f], index=feat_df.index, dtype=np.float64)

                if 'std' in self.modes:
                    current_std = groupby_feature[self.target].loc[train_index].std()
                    feat_df = ((current_std * current_size + global_std * self.alpha) / (current_size + self.alpha)).fillna(global_std)
                    self.stds_train[fold][f] = pd.DataFrame(feat_df, columns=["%s_stds_SA" % f], index=feat_df.index, dtype=np.float64)
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
                self.counts_train[f] = pd.DataFrame(current_size_test, columns=["%s_counts_SA" % f], index=current_size_test.index, dtype=np.float64)

            if 'mean' in self.modes:
                current_mean_test = groupby_feature_test[self.target].mean()
                feat_df_test = ((current_mean_test * current_size_test + self.global_mean_test * self.alpha) / (current_size_test + self.alpha)).fillna(self.global_mean_test)
                self.mean_test[f] = pd.DataFrame(feat_df_test, columns=["%s_mean_SA" % f], index=feat_df_test.index, dtype=np.float64)

            if 'std' in self.modes:
                current_std_test = groupby_feature_test[self.target].std()
                feat_df_test = ((current_std_test * current_size_test + self.global_std_test * self.alpha) / (current_size_test + self.alpha)).fillna(self.global_std_test)
                self.stds_test[f] = pd.DataFrame(feat_df_test, columns=["%s_stds_SA" % f], index=feat_df_test.index, dtype=np.float64)

    
    def transform(self, df, mode='train'):
        try:
            df.drop([i for i in df.columns if re.search('_SA', i)], axis=1, inplace=True)
        except:
            pass
        _train, _test = [], []
        if 'counts' in self.modes:
            _train.append(self.counts_train)
            _test.append(self.counts_test)
        if 'mean' in self.modes:
            _train.append(self.mean_train)
            _test.append(self.mean_test)
        if 'std' in self.modes:
            _train.append(self.stds_train)
            _test.append(self.stds_test)
        for values_train, values_test, name in zip(_train, _test, self.modes):
            lvl2 = None
            if mode == 'train':
                fold = 1
                for train_index, test_index in self.cv.split(df):
                    if name == 'mean':
                        global_ = df[self.target].loc[train_index].mean()
                    if name == 'std':
                        global_ = df[self.target].loc[train_index].std()
                    lvl1 = None
                    for f in self.features:
                        if lvl1 is not None:
                            lvl1 = pd.merge(lvl1, values_train[fold][f], how="left", left_on=f, right_index=True)
                        else:
                            lvl1 = pd.merge(df.loc[time], values_train[fold][f], how="left", left_on=f, right_index=True)
                        if name != 'counts':
                            lvl1['%s_%s_SA' % (f, name)] = lvl1['%s_%s_SA' % (f, name)].fillna(global_)
                        else:
                            lvl1['%s_%s_SA' % (f, name)] = lvl1['%s_%s_SA' % (f, name)]
                    if lvl2 is not None:
                        lvl2 = pd.concat([lvl2, lvl1], axis=0, ignore_index=False, join='outer', copy=False)
                    else:
                        lvl2 = lvl1
                    fold += 1
                df = lvl2
                del lvl1
                del lvl2
            elif mode == 'test':
                for f in self.features:
                    df = pd.merge(df, values_test[f], how="left", left_on=f, right_index=True)
                    df['%s_%s_SA' % (f, name)] = df['%s_%s_SA' % (f, name)].fillna(self.global_mean_test)
        df.drop_duplicates(subset=['account id'], keep='last', inplace=True)
        return df