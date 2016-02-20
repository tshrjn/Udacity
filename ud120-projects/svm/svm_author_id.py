#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

from sklearn import svm

clf = svm.SVC(kernel="rbf",C=10000)
    # ,gamma=1000,C=100000)

### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
### return the fit classifier
t1 = time()
pred = clf.predict(features_test)
print "Predicting time:", round(time()-t1, 3), "s"
# return clf
### your code goes here!
from sklearn.metrics import accuracy_score
print ("After Classifying, the accuracy of SVM author identifier is:")
print accuracy_score(pred,labels_test)
print " predict for element 10 of the test set? The 26th? The 50th? ",pred[10],pred[26],pred[50]

count = 0
for ii in pred:
    if ii == 1:
        count += 1
print "No of predicted Chris's emails are: ",count

#########################################################


