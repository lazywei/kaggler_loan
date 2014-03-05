#
#
#Beating the Benchmark :::::: Kaggle Loan Default Prediction Challenge.
#__author__ : Abhishek
#
# python loan_svm.py <traindata> <testdata> <outputfile> <regression_type>

import pandas as pd
import numpy as np
import cPickle
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import  scipy.stats as stats
import sklearn.linear_model as lm
import sklearn.ensemble as ens 
from sys import argv

def main():

 X, labels = traindata(argv[1])
 X_test = testdata(argv[2])

 
 X = preprocessing.scale(X) 
 X_test = preprocessing.scale(X_test)

 createSub( X, labels, X_test)


def testdata(filename):
 X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)

 X = np.asarray(X.values, dtype = float)

# col_mean = stats.nanmean(X,axis=0)
# inds = np.where(np.isnan(X))
# X[inds]=np.take(col_mean,inds[1])
 data = np.asarray(X[:,1:-3], dtype = float)

 return data
 
def traindata(filename):
 X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)

 X = np.asarray(X.values, dtype = float)

# col_mean = stats.nanmean(X,axis=0)
# inds = np.where(np.isnan(X))
# X[inds]=np.take(col_mean,inds[1])

 labels = np.asarray(X[:,-1], dtype = float)
 data = np.asarray(X[:,1:-4], dtype = float)
 return data, labels


def createSub( traindata, labels, testdata):
 
  labels = np.asarray(map(int,labels))
 
  xtrain = traindata#[train]
  xtest = testdata#[test]

  ytrain = labels#[train]
  predsorig_train = np.asarray([0] *traindata.shape[0]) #np.copy(ytest)
  predsorig_test = np.asarray([0] * testdata.shape[0]) #np.copy(ytest)

  labelsP =np.asarray(map(lambda x: 1 if x > 0 else 0,labels))
  ytrainP = labelsP

  #http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, verbose = 2)
  lsvc.fit(xtrain, ytrainP)
    
  xtrainP = lsvc.transform(xtrain)
  xtestP =  lsvc.transform(xtest)

  clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1.0, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)
 #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  clf2 = ens.GradientBoostingRegressor(loss='quantile', alpha=0.95,
                              n_estimators=250, max_depth=3,
                              learning_rate=.1, min_samples_leaf=9,
                              min_samples_split=9)
 #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

 #=Regression 1================================
  clf.fit(xtrainP,ytrainP)
  predsP = clf.predict(xtestP)
 #=============================================
  nztrain = np.where(ytrainP > 0)[0]
  nztest = np.where(predsP == 1)[0]

  nztrain0 = np.where(ytrainP == 0)[0]
  nztest0 = np.where(predsP == 0)[0]

  xtrainP = xtrain[nztrain]
  xtestP = xtest[nztest]

  ytrain0 = ytrain[nztrain0]
  ytrain1 = ytrain[nztrain]

  regression_type=argv[4]
#=Regression 2================================
  if regression_type=="logistic":
      print "logistic regression"
      clf.fit(xtrainP,ytrain1)
      preds_train= clf.predict(xtrainP)
      preds_test = clf.predict(xtestP)
      predsorig_train[nztrain]= preds_train
      predsorig_test[nztest] = preds_test
#=============================================
  elif regression_type=="quantile":
      print "quantile regression"
      clf2.fit(xtrainP,ytrain1)
      preds_train= clf2.predict(xtrainP)
      preds_test= clf2.predict(xtestP)
      predsorig_train[nztrain] = np.asarray(map(int,preds_train))
      predsorig_test[nztest] = np.asarray(map(int,preds_test))
#=============================================
  else:
      print "error: wrong regression type"
      return

  #print np.sum(predsorig)
  #predsorig[nztest0] = 0
  #print np.sum(predsorig)

  output_filename=argv[3].replace(".csv","")
  np.savetxt("%s_train.csv"%(output_filename),predsorig_train ,delimiter = ',', fmt = '%d')
  np.savetxt("%s_test.csv"%(output_filename),predsorig_test ,delimiter = ',', fmt = '%d')

if __name__ == '__main__':
    if len(argv) < 5:
        print "loan_quantile.py <filename train> <filename test> <filename result> <regression_type>"
    else:
        main()
