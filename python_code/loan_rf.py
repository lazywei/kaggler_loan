import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing

train_data = np.loadtxt('../cleaned_data/cleaned_train.csv', delimiter=',', skiprows=1)
test_data  = np.loadtxt('../cleaned_data/cleaned_test.csv', delimiter=',', skiprows=1)

train_x   = preprocessing.scale(train_data[:, 0:-1])
train_y   = train_data[:, -1]
test_data = preprocessing.scale(test_data)

# Index of non zero y
nz_ind_on_train = np.where(train_y != 0)[0]
train_y_for_clf = train_y.astype(int)

train_y_for_clf[nz_ind_on_train] = 1

clf = RandomForestClassifier(n_estimators=100, verbose=1)
clf = clf.fit(train_x, train_y_for_clf)

pred_on_test = clf.predict(test_data)

nz_ind_on_test = np.where(pred_on_test != 0)[0]

rgs = RandomForestRegressor(n_estimators=100, verbose=1)
rgs = rgs.fit(train_x[nz_ind_on_train,], train_y[nz_ind_on_train])

pred_on_test[nz_ind_on_test] = rgs.predict(test_data[nz_ind_on_test,])

np.savetxt("rf_out.csv", pred_on_test, delimiter = ',')
