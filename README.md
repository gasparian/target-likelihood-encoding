# target-likelihood-encoding

Basic idea: let's use target values to compute statistics for every sub-category of categorical features according to this formula:  

*smoothed likelihood* = (*fold_target_stat* * nrows + *global_val* * *alpha*) / (*nrows* + *alpha*)  

where:  
* *target_statistic* - target statistic value across current fold,
* *global_val* - target statistic value across all train set, 
* *alpha* - regularization value.  

So if we have a rare subclass, it's target statistic will tend to the global value.

See the code for more info.  

## Usage

This kind of features leads to overfitting, so it must be created **inside** the cross-validation loop.  

```
encoding = TLEncoding(n_splits=10, alpha=10, target='conversion')
encoding.fit(train)
tl_train = encoding.transform(train, mode='train')
tl_test = encoding.transform(test, mode='test')
```

## Dependencies  
* python 3.6
* numpy 1.12.1
* pandas 0.20.1 
