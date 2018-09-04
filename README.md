# Likelihood-Counts features

This algorithm is very helpful for handling the categorical features. It uses target values to compute target statistics for every sub-category according to this formula:  

*smoothed likelihood* = (*fold_target_stat* * nrows + *global_val* * *alpha*) / (*nrows* + *alpha*)  

where:  
* *target_statistic* - target statistic value across current fold,
* *global_val* - target statistic value across all train set, 
* *alpha* - regularization value.  

So if we have a rare subclass, it's target statistic will tend to the global value.

See the code for more info.  

## Installation
```
pip install lcfeatures
```

## Usage

This kind of features leads to overfitting, so LC-Features must be created **inside** the cross-validation loop.  

```
encoding = LCfeatures(TimeSeriesSplit(n_splits=5), modes=['mean'], alpha=10, features='all', target='conversion'))
encoding.fit(train)
train = encoding.transform(train, mode='train')
test = encoding.transform(test, mode='test')
```
After that, new columns with suffix "_LC" will be created.

## Dependencies  
* python 3.6
* numpy 1.12.1
* pandas 0.20.1 
