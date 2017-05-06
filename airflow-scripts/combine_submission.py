import itertools
import numpy as np
import pandas as pd
from math import exp, log
import sys

# files=3 w1=0.1 w2=0.2 w3=0.7 f1=temp.csv f2=temp2.csv f3=temp3.csv output=filename.csv
# python combine_submission.py files=3 w1=0.05 w2=0.47 w3=0.48 f1=predictions_1_way.csv f2=predictions_keras.csv f3=predictions_2_way.csv output=FRIENDS.csv
print("Script name:",sys.argv[0])
args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
print(args)

file_count = int(args['files'])
total_weight = 0.0
for i in range(1,file_count+1):
	print('w'+str(i))
	total_weight=total_weight + float(args['w'+str(i)])
if int(total_weight) != 1:
	print("All weights must sum upto 1. Please re-run the file.")
	quit()
else:
	print("All weights sum upto 1")
	for i in range(1,file_count+1):
		if i==1:
			submission = pd.read_csv(args['f1'])
			submission['loss'] *= float(args['w1'])
		else:
			submission['loss'] += float(args['w'+str(i)]) * pd.read_csv(args['f'+str(i)])['loss'].values

	submission.to_csv(args['output'],index=False)
	print("File written successfully")
'''
submission = pd.read_csv('predictions_1_way.csv')
submission['loss'] *= 0.05
submission['loss'] += 0.47 * pd.read_csv('predictions_keras.csv')['loss'].values
# prediction from https://www.kaggle.com/mariusbo/allstate-claims-severity/lexical-encoding-feature-comb-lb-1109-05787
submission['loss'] += 0.48 * pd.read_csv('predictions_2_way.csv')['loss'].values
submission.to_csv('combine_submission_new.csv', index=False)

'''
'''
import pandas as pd
# prediction from https://www.kaggle.com/iglovikov/allstate-claims-severity/xgb-1114
submission = pd.read_csv('predictions_1_way.csv')
submission['loss'] *= 0.05
submission['loss'] += 0.37 * pd.read_csv('predictions_keras.csv')['loss'].values
# prediction from https://www.kaggle.com/mariusbo/allstate-claims-severity/lexical-encoding-feature-comb-lb-1109-05787
submission['loss'] += 0.58 * pd.read_csv('predictions_2_way.csv')['loss'].values
submission.to_csv('combine_submission_3_way_2.csv', index=False)
'''
