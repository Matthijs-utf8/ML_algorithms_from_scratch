# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:00:24 2019

@author: Matthijs Schrage
"""
"""
Euclidean distance is the distance from one (test) data point to all the (train) data points
It is calculated by squaring (the sum of (the difference of the coordinates of a certain data type))
It looks like: sqrt(sum(a_i - p_i)**2) for every i in the data set (i are the dimensions)
a_i in this case is a poit in the data set available, p_i is the point you want to test
"""

import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=5):
    #Check if the user has used the right amount of datagroups
    if len(data) >= k:
        warnings.warn('K is set to a value less then total voting groups!')
    
    #Calculate the distances from the the point to all other data points and 
    #note the datagroup of all the points together with the distance
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    
    #Calculating the vote for the predicted group
    votes =[i[1] for i in sorted(distances)[:k]]
#    print(distances)
    vote_result = Counter(votes).most_common(1)[0][0] #Sort the votes and get the most common vote
    
    #Creating a value for the confidence of the prediction based on the vote
    confidence = Counter(votes).most_common(1)[0][1] / k
    
    return vote_result, confidence

#Reading the data
df = pd.read_csv('breast-cancer-wisconsin.data')

#Replacing all question marks (which are spots that are missing data) with -99999 so the computer can read them as an outliër
df.replace('?', -99999, inplace=True)

#Dropping the id column because it has irrelevant information
df.drop( ['id'], 1, inplace=True)

#Making every value in the dataset a float type number
full_data = df.astype(float).values.tolist()

#Shuffle the data so now it is a new dataset
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))] #the first 80% of the data
test_data = full_data[-int(test_size*len(full_data)):]#the last 20% of the data

#print(train_data)
for i in train_data:
    train_set[i[-1]].append(i[:-1]) #take out the class column, get all the data except the last column
#print(train_set)
    
for i in test_data:

    test_set[i[-1]].append(i[:-1]) #take out the class column, get all the data except the last column

correct = 0
total = 0

for group in test_set:
   print(group)
   for data in test_set[group]:
       vote, condfidence = k_nearest_neighbors(train_set, data, k=5)
       if group == vote:
           correct += 1
       else:
           print(condfidence) #print the confidence of the data the algorithm got wrong
           pass
       total += 1


print('Accuracy: ' + str(correct/total))
