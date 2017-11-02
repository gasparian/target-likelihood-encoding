# Likelihood-Counts features

This algorithm is very helpful for handling the categorical features. It uses target values to compute smoothed likelihood (and counts) for every sub-category according to this formula:  

*smoothed likelihood* = (*mean*(*target*) * nrows + *global mean* * *alpha*) / (*nrows* + *alpha*)  

where:  
* *global mean* - average target value across all train set, 
* *alpha* - regularization.  

So if we have a rare subclass, it's likelihood will tend to the global mean value.

See the code for more info.  

## Installation
```
pip instasll lcfeatures
```

## Usage

This kind of features leads to overfitting, so LC-Features must be created **inside** the cross-validation loop.  

```
encoding = LCfeatures(TimeSeriesSplit(n_splits=5), modes=['mean', 'std', 'counter'], alpha=10, features='all', target='conversion'))
encoding.fit(train)
train = encoding.transform(train, mode='train')
test = encoding.transform(test, mode='test')
```

## Dependencies  
* python 3.6
* numpy 1.12.1
* pandas 0.20.1 
