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
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
import sklearn.linear_model as lm
import sklearn.ensemble as ens
import argparse

def main(args):
  train_file      = args.train_file
  test_file       = args.test_file
  output_filename = args.output_file.replace(".csv", "")
  regression_type = args.regression_type
  cv_mode         = args.cv_mode
  logregC         = args.logregC
  lsvcC           = args.lsvcC

  # train_file      = "../cleaned_data/cleaned_train.csv"
  # test_file       = "../cleaned_data/cleaned_test.csv"
  # regression_type = 'logistic'
  # output_filename = "out.csv".replace(".csv","")

  X, labels = traindata(train_file)
  X_test    = testdata(test_file)

  X         = preprocessing.scale(X)
  X_test    = preprocessing.scale(X_test)

  if cv_mode:
    permuted_index = np.random.permutation(X.shape[0])

    k_fold = 3

    mae = 0
    for val_index in np.array_split(permuted_index, k_fold):
      train_index = np.delete(permuted_index, val_index)

      train_data  = X[train_index,]
      train_label = labels[train_index,]

      val_data    = X[val_index,]
      val_label   = labels[val_index,]

      _, pred_on_val = trainer(train_data, train_label, val_data, regression_type, lsvcC, logregC)
      mae = mae + mean_absolute_error(val_label, pred_on_val)
    print "MAE:" + str(mae/k_fold)
  else:
    pred_on_train, pred_on_test = trainer(X, labels, X_test, regression_type, lsvcC, logregC)
    print "In sample MAE:" + str(mean_absolute_error(pred_on_train, labels))
    createSub(pred_on_train, pred_on_test, output_filename)

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

def trainer(traindata, labels, testdata, regression_type, lsvcC=0.01, logregC=1.0):
  labels = np.asarray(map(int,labels))

  xtrain = traindata#[train]
  xtest  = testdata#[test]
  ytrain = labels#[train]

  predsorig_train = np.asarray([0] *traindata.shape[0]) #np.copy(ytest)
  predsorig_test  = np.asarray([0] * testdata.shape[0]) #np.copy(ytest)

  labelsP = np.asarray(map(lambda x: 1 if x > 0 else 0,labels))
  ytrainP = labelsP

  #http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  lsvc = LinearSVC(C=lsvcC, penalty="l1", dual=False, verbose = 2)
  lsvc.fit(xtrain, ytrainP)

  xtrainP = lsvc.transform(xtrain)
  xtestP  = lsvc.transform(xtest)

  clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                             C=logregC, fit_intercept=True, intercept_scaling=1.0,
                             class_weight=None, random_state=None)
  #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  clf2 = ens.GradientBoostingRegressor(loss='quantile', alpha=0.5,
                              n_estimators=250, max_depth=3,
                              learning_rate=.1, min_samples_leaf=9,
                              min_samples_split=9)
  #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

  #=Regression 1================================
  clf.fit(xtrainP,ytrainP)
  predsP = clf.predict(xtestP)
  #=============================================
  nztrain  = np.where(ytrainP > 0)[0]
  nztest   = np.where(predsP == 1)[0]

  nztrain0 = np.where(ytrainP == 0)[0]
  nztest0  = np.where(predsP == 0)[0]

  xtrainP  = xtrain[nztrain]
  xtestP   = xtest[nztest]

  ytrain0  = ytrain[nztrain0]
  ytrain1  = ytrain[nztrain]

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

  return predsorig_train, predsorig_test


def createSub(predsorig_train, predsorig_test, output_filename):
  np.savetxt("%s_train.csv"%(output_filename),predsorig_train ,delimiter = ',', fmt = '%d')
  np.savetxt("%s_test.csv"%(output_filename),predsorig_test ,delimiter = ',', fmt = '%d')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Predict loan default by logistic/quantile regression.')
  parser.add_argument('train_file', metavar='train_file', type=str, help='train file path')
  parser.add_argument('test_file', metavar='test_file', type=str, help='test file path')
  parser.add_argument('output_file', metavar='output_file', type=str, help='output file path')
  parser.add_argument('regression_type', metavar='regression_type', type=str, choices=['logistic', 'quantile'])
  parser.add_argument('--lsvcC', metavar='lsvcC', type=float)
  parser.add_argument('--logregC', metavar='logregC', type=float)
  parser.add_argument('--cv', dest='cv_mode', help='use CV mode (no prediction on test data will be made)', action='store_true')
  parser.add_argument('--no-cv', dest='cv_mode', help='use CV mode (no prediction on test data will be made)', action='store_false')
  parser.set_defaults(cv_mode=False)
  parser.set_defaults(lsvcC=0.01)
  parser.set_defaults(logregC=1)
  args = parser.parse_args()

  print args

  main(args)
