import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew, boxcox
from math import exp, log

from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

#reading data
shift=200
train_load = pd.read_csv('train.csv')
test_load = pd.read_csv('test.csv')
train_labels = np.log(train_load['loss']+shift).ravel()
test_indx=test_load['id']
train_indx=train_load['id']

#combine train-test
test_load['loss'] = np.nan
joined = pd.concat([train_load, test_load]).reset_index(drop=True)
joined = joined.drop(['id','loss'],1)

cat_feature = [n for n in joined.columns if n.startswith('cat')]    
cont_feature = [n for n in joined.columns if n.startswith('cont')] 

#factorize categorical features
for column in cat_feature:
        joined[column] = pd.factorize(joined[column].values, sort=True)[0]

# for continuous features: compute skew and do Box-Cox transformation
skewed_feats = train_load[cont_feature].apply(lambda x: skew(x.dropna()))
print("\nSkew in numeric features:")
print(skewed_feats)
# transform features with skew > 0.25 (this can be varied to find optimal value)
skewed_feats = skewed_feats[skewed_feats > 0.25]
skewed_feats = skewed_feats.index
for feats in skewed_feats:
	joined[feats] = joined[feats] + 1
	joined[feats], lam = boxcox(joined[feats])

x_train = joined.iloc[:train_load.shape[0], :]
x_test = joined.iloc[train_load.shape[0]:, :]

joined_scaled, scaler = scale_data(joined)
train, _ = scale_data(x_train, scaler)
test, _ = scale_data(x_test, scaler)

train = pd.DataFrame(train)
train = train.assign(labels = train_labels)
train = train.assign(id = train_indx)
test =	pd.DataFrame(test)
test = test.assign(id=test_indx)
train.to_csv("train_processed.csv",index=False)
test.to_csv("test_processed.csv",index=False)