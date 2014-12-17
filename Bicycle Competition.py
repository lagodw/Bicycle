"""
Created on Mon Dec 15 19:03:23 2014

@author: lagodw
"""

''' Code for kaggle competition to predict public bike rental in DC's Capital Bikeshare program
The model used is a stacking of a decision tree and k nearest neighbors, both of which performed well on their own
The training data used included hourly rental rates for the first 19 days of each month in 2011 and 2012 with the test data including the missing days
The data included information on the date, season, and weather '''

import csv
import numpy as np
import pandas as pd
from sklearn import tree
from math import log
from math import sqrt
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn.ensemble import ExtraTreesClassifier



traindata = pd.read_csv('trainbike.csv', index_col=0, parse_dates=True)
testdata = pd.read_csv('testbike.csv', index_col=0, parse_dates=True)

# Separating the features from the outcomes
features = traindata.iloc[:,:-3] 
featurestest = testdata.iloc[:,:-3]

# Extracting the date and time from the index
features['weekday'] = features.index.weekday
features['hour'] = features.index.hour
features['year'] = features.index.year

featurestest['weekday'] = featurestest.index.weekday
featurestest['hour'] = featurestest.index.hour
featurestest['year'] = featurestest.index.year


# Index to be used for writing output
testindex = testdata.index

'''The features included a number of metrics on weather, many of which seemed redundant
After feature selection methods, weather and temperature were the most important features
Rental counts were much higher during the morning and afternoon and during the middle of the week. There also appeared to be large yearly and seasonal effects
Count is the dependent variable, the number of bicycles rented each hour
Registered and casual are divisions of the rental counts, with registered + casual = count '''

weathers = np.transpose(np.array([features['weather'],features['temp'],features['atemp'],features['humidity'],features['windspeed']]))

count = np.array(traindata.iloc[:,-1])
registered = np.array(traindata.iloc[:,-3])
casual = np.array(traindata.iloc[:,-2])

clf = ExtraTreesClassifier()
clf.fit(weathers,count)
print clf.feature_importances_

features = np.transpose(np.array([features['year'],features['hour'],features['weekday'],features['season'],traindata['weather'],traindata['temp']]))
featuresval = np.transpose(np.array([featurestest['year'],featurestest['hour'],featurestest['weekday'],featurestest['season'],featurestest['weather'],featurestest['temp']]))


# Spliting the data for cross validation
featuretrain, featuretest, counttrain, counttest, registeredtrain, registeredtest, casualtrain, casualtest = train_test_split(features,count,registered,casual,test_size=0.3,random_state=0)

del traindata
del testdata

# The decision tree, which interestingly performed better than random forests for this problem

tree = tree.DecisionTreeClassifier()
predicttree = tree.fit(featuretrain, counttrain).predict(featuretest)

# The competition used a Root Mean Squared Logarithmic Error, calculated for each model here

def MSLE(p,a):
    return (log(p+1) - log(a+1))**2
    
scoretree = 0
for i in range(len(predicttree)):
    scoretree += MSLE(predicttree[i],counttest[i]) 
scoretree = sqrt(scoretree / len(predicttree))
print 'Decision Tree score = ' + str(scoretree)

# The K nearest neighbors model, k=7 appeared to perform best

nn = neighbors.KNeighborsClassifier(7, weights='distance')
predictneighbor = nn.fit(featuretrain, counttrain).predict(featuretest)
    
scoreneighbor = 0
for i in range(len(predictneighbor)):
    scoreneighbor += MSLE(predictneighbor[i],counttest[i]) 
scoreneighbor = sqrt(scoreneighbor / len(predictneighbor))
print 'Nearest Neighbor score = ' + str(scoreneighbor)

'''The stacked model uses an average of the two model's predictions, with weights assigned to each model based on their relative cross evalutaion performance'''

# The weight for the tree model
treeweight = scoretree / (scoretree + scoreneighbor)

# The stacked testing split predictions and the score of the stacked model, which performed better than either model individually
stacked = treeweight * predicttree + (1-treeweight) * predictneighbor

scorestacked = 0
for i in range(len(stacked)):
    scorestacked += MSLE(stacked[i],counttest[i]) 
scorestacked = sqrt(scorestacked / len(stacked))
print 'Stacked score = '+ str(scorestacked)


# Generating the stacked predictions to be used in the competition
predict = treeweight * tree.fit(features,count).predict(featuresval) + (1-treeweight) * nn.fit(features, count).predict(featuresval)


'''The previous models have been predicting the total count of bicycles rented
Now I will predict the casual and registered rentals separately and add them together to estimate total rentals
This might allow for more accurate predictions of each and therefore eliminate some of the noise in total rentals'''

predicttreeregistered = tree.fit(featuretrain, registeredtrain).predict(featuretest)
predictneighborregistered = nn.fit(featuretrain, registeredtrain).predict(featuretest)

predicttreecasual = tree.fit(featuretrain, casualtrain).predict(featuretest)
predictneighborcasual = nn.fit(featuretrain, casualtrain).predict(featuretest)

predicttreecombined = predicttreeregistered + predicttreecasual
predictneighborcombined = predictneighborregistered + predictneighborcasual

scoretreenew = 0
for i in range(len(predicttreecombined)):
    scoretreenew += MSLE(predicttreecombined[i],counttest[i]) 
scoretreenew = sqrt(scoretreenew / len(predicttreecombined))
print 'Combined tree score = ' + str(scoretreenew)

scoreneighbornew = 0
for i in range(len(predictneighborcombined)):
    scoreneighbornew += MSLE(predictneighborcombined[i],counttest[i]) 
scoreneighbornew = sqrt(scoreneighbornew / len(predictneighborcombined))
print 'Combined neighbors score = ' + str(scoreneighbornew)

treeweight = scoretree / (scoretree + scoreneighbor)
stacked = treeweight * predicttree + (1-treeweight) * predictneighbor


stackedcombined = treeweight * predicttreecombined + (1-treeweight) * predictneighborcombined


scorestackednew = 0
for i in range(len(stackedcombined)):
    scorestackednew += MSLE(stackedcombined[i],counttest[i]) 
scorestackednew = sqrt(scorestackednew / len(stackedcombined))
print 'Combined stacked score = ' + str(scorestackednew)


predictcombined = treeweight * (tree.fit(features,casual).predict(featuresval) + tree.fit(features,registered).predict(featuresval)) + (1-treeweight) * (nn.fit(features, casual).predict(featuresval) + nn.fit(features, registered).predict(featuresval))


# Writing the csv file    
predictionfile = open('stackedcombined.csv', 'wb')
predictionfileobject = csv.writer(predictionfile)

predictionfileobject.writerow(['datetime','count'])
for row in range(len(featuresval)):
    index = str(testindex[row])
    predictionfileobject.writerow([index,predictcombined[row]])
predictionfile.close()




