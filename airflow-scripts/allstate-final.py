import itertools
import numpy as np
import pandas as pd
from math import exp, log
import sys

print("Setting up models...")
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# python allstate-final.py classifier=rfr
train = pd.read_csv('train_processed.csv')
test = pd.read_csv('test_processed.csv')
train_labels= train['labels'].ravel()
test_id = test['id']
train_id = train['id']
train = train.drop(['id','labels'],axis=1)
test = test.drop('id',axis=1)
shift = 200

print("Script name:",sys.argv[0])
args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
print(args)


ESTIMATORS = {
	"encv":    ElasticNetCV(),
	"rfr":     RandomForestRegressor(n_estimators=250),
	"svr":     SVR(C=1.0, epsilon=0.2),
	"gbr":     GradientBoostingRegressor(n_estimators=250),
	"adb":     AdaBoostRegressor(n_estimators=250),
	"knn4":   KNeighborsRegressor(n_neighbors=4)
}

test_predictions = pd.DataFrame({'id': test_id, 'loss': np.nan})
test_predictions.set_index(['id'])

name = args['classifier']
output = args.get("output",name+'_predictions.csv')

if name in ESTIMATORS.keys():
	estimator = ESTIMATORS[name]
	estimator.fit(train, train_labels)
	test_labels = np.exp(estimator.predict(test))-shift
	test_predictions = test_predictions.assign(loss = test_labels)
	test_predictions.to_csv(output, index=False)
	print("Model: ", name,"output file name: ",output)