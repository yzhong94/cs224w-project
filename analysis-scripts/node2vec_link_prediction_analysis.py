import snap
import pandas as pd
import link_prediction
import node2vec_link_prediction_feature_extraction
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

def main():
    '''
    run node2vec_link_prediction_analysis first
    '''
    
    p = 1
    q = 2

    print "looping p",p
    print "looping q",q
    X, Y = node2vec_link_prediction_feature_extraction.getXYFromEmb(bill_term = 100,fin_start_year = 1985,fin_end_year = 1986,p = p,q = q)
    X_test, Y_test = node2vec_link_prediction_feature_extraction.getXYFromEmb(bill_term = 101,fin_start_year = 1987,fin_end_year = 1988,p = p,q = q)

    X.to_csv('X_emb.csv', index = False)
    Y.to_csv('Y_emb.csv', index = False)

    X_test.to_csv('X_test.csv', index = False)
    Y_test.to_csv('Y_test.csv', index = False)
    
    X = pd.read_csv('X_emb.csv')
    Y = pd.read_csv('Y_emb.csv')

    X_test = pd.read_csv('X_test.csv')
    Y_test = pd.read_csv('Y_test.csv')
    
    print "baseline", Y[Y['result'] == 1].shape[0]/float(Y.shape[0])
    Y = Y['result']
    Y_test = Y_test['result']

    print X.describe()
    print Y.describe()
    inds = pd.isnull(X).any(1).nonzero()[0]

    X =  X.drop(inds)
    Y =  Y.drop(inds)
    
    inds = pd.isnull(X_test).any(1).nonzero()[0]

    X_test =  X_test.drop(inds)
    Y_test=  Y_test.drop(inds)

    print "---with no feature selection---"
    clf = link_prediction.getlogistic(X,Y)
    print clf.score(X_test,Y_test)
    
    print "---with 20 perc select percentile---"
    selector = SelectPercentile(f_classif, percentile=20)
    selector.fit(X, Y)
    clf = link_prediction.getlogistic(selector.transform(X),Y)
    print clf.score(selector.transform(X_test),Y_test)

    '''
    print "---with select percentile---"
    perc_scores_train = []
    perc_scores_test = []
    for i in range(1,50,5):
        selector = SelectPercentile(f_classif, percentile=i)
        selector.fit(X, Y)
        clf = link_prediction.getlogistic(selector.transform(X),Y)
        print clf.score(selector.transform(X_test),Y_test)
        perc_scores_test.append(clf.score(selector.transform(X_test),Y_test))
    print perc_scores_test
    '''
    
    '''
    print "---with RFE CV---"
    perc_scores_train = []
    perc_scores_test = []
    for i in range(10,50,10):
        print i
        estimator = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
        #estimator = link_prediction.getlogistic(selector.transform(X),Y)
        selector = RFECV(estimator, step=1, cv=2)
        selector.fit(X, Y)
        print clf.score(selector.transform(X_test),Y_test)
        perc_scores_test.append(clf.score(selector.transform(X_test),Y_test))
    print perc_scores_test
    '''
    pass

if __name__ == "__main__":
    main()