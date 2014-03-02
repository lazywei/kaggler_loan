#
#
#Beating the Benchmark :::::: Kaggle Loan Default Prediction Challenge.
#__author__ : Abhishek
#
#

import numpy as np
from sklearn import preprocessing
from sys import argv
import sklearn.linear_model as lm

def main():
    X_train, labels = traindata(argv[1])
    X_test = testdata(argv[2])
    shape_train=X_train.shape[0]
   # shape_test=X_test.shape[0]
    X_all= preprocessing.scale(np.concatenate((X_train,X_test), axis=0))
    #print shape_train,shape_test 
    X_train = X_all[:shape_train,:]
    X_test = X_all[shape_train:,:]
   # print X_train.shape[0],X_test.shape[0]

    createSub( X_train, labels, X_test)

def traindata(filename):
    data_raw=testdata(filename)
    labels = data_raw[:,-1]    
    data = data_raw[:,:-1]
    return data, labels

def testdata(filename):
    f=open(filename,'r')
    data=np.asarray([ l.strip().split(',') for l in f.readlines() ],dtype=float)
    f.close()
    return data
 

#def createSub( xtrain, labels, xtest):
def createSub(xtrain, labels,xtest):

    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1.0, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

    labels = np.asarray(map(int,labels))


    ytrain = labels#[train]
    predsorig_test= np.asarray([0] * xtest.shape[0]) #np.copy(ytest)
    predsorig_train = np.asarray([0] * xtrain.shape[0]) #np.copy(ytest)

    labelsP =np.asarray(map(lambda x: 1 if x > 0 else 0,labels))
    ytrainP = labelsP
    xtrainP = xtrain
    xtestP =  xtest
    #print ytrain

    #======Regression 1===========================
    clf.fit(xtrainP,ytrainP)
    predsP = clf.predict(xtestP)
    #=============================================

    nztrain = np.where(ytrainP > 0)[0]
    nztest = np.where(predsP > 0)[0]

    xtrainP = xtrain[nztrain]
    xtestP = xtest[nztest]

    ytrain1 = ytrain[nztrain]

    #======Regression 2===========================
    clf.fit(xtrainP,ytrain1)
    preds_train= clf.predict(xtrainP)
    preds_test= clf.predict(xtestP)
    #=============================================

    predsorig_train[nztrain]= preds_train
    #print preds_train
    predsorig_test[nztest] = preds_test
    output_filename=argv[3].replace(".csv","")
    np.savetxt(("%s_train.csv")%(output_filename),predsorig_train,delimiter = ',', fmt = '%d')
    np.savetxt(("%s_test.csv")%(output_filename),predsorig_test,delimiter = ',', fmt = '%d')

if __name__ == '__main__':
    if len(argv) < 4:
        print "loan_logistic.py <filename train> <filename test> <filename result>"
    else:
        main()
