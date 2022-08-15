from AutoEn import AutoEn

import pandas as pd 
import numpy as np
import sys


missing_values = ["n/a", "na", "--", "?"]
df = pd.read_csv('{}.csv'.format(sys.argv[1]), na_values=missing_values)
x_cols = [c for c in df.columns if c != 'class']
X = df[x_cols]
y = df['class']
print(X.shape, y.shape)
X.fillna(X.mean(), inplace=True)
XX = pd.get_dummies(X, prefix_sep='_', drop_first=True)



clf = AutoEn()
clf.fit(XX,y)
clf.predict(XX)
clf.predict_proba(XX)


