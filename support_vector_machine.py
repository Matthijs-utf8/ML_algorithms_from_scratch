# -*- coding: utf-8 -*-
"""
Created on Fri May 17 22:20:44 2019

@author: Matthijs Schrage
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    #If you just call the support_vector_machine class to run, just the ini method wil run, other methods you have to specify
    def __init__(self, visualization=True):
        #If the user doesnt specify visualization, it is True
        self.visualization = visualization
        #The colors of the negative class is blue, the positive one is red
        self.colors = {1:'r', -1:'b'}
        #If you want to visualize the data, python plots the data
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    #Train the algorithm with the data
    def fit(self, data):
        self.data = data
        
        opt_dict = {}
        
        #Transform w to check every posibility
        transforms = [[1,1],[-1,1],[1,-1],[-1,-1]]
        
        all_data = []
        #For every class (yi) in the data, so '+' or '-'.
        for yi in self.data:
            #For every featureset in the class '-' or '+'.
            for featureset in self.data[yi]:
                #For every feature in the featureset. The features can be seen as dimensions
                for feature in featureset:
                    #Add the data to all_data
                    all_data.append(feature)
        #Get the max and min value from all the data
        self.max_feature_value = max(all_data) 
        self.min_feature_value = min(all_data)
        all_data = None
        
        #We approximate vector w in smaller and smaller stepsizes
        step_sizes = [self.max_feature_value * 0.1, 
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,]
        
        #Changing the stepsize for b because it does not need to be as big as the w stepsize does
        b_range_multiple = 5
        b_multiple = 5
        #Set the first value of w at 10 times the maximum value of all_data
        latest_optimum = self.max_feature_value*10
        
        #Going through every possiblities for w and b
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum]) #Start the vector w at these values
            optimized = False
            while not optimized: #Solving the convex problem
                #Finding a 'b' vector
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), #Min of the range
                                   self.max_feature_value*b_range_multiple, #Max of the range
                                   step*b_multiple): #Stepsize 5 times bigger than the w stepsize
                    #Finding a 'w' vector
                    for transformation in transforms: #Check 'w' for every possible transformation
                        w_transform = w * transformation
                        found_option = True
                        for yi in self.data: #For class in the data, '-' or '+'
                            for xi in self.data[yi]: #For feature in the class
                                if not yi *(np.dot(w_transform,xi)+b)>=1: #yi(xi.w+b) >= 1
                                    found_option = False
                        
                        if found_option:
                            opt_dict[np.linalg.norm(w_transform)] = [w_transform,b]
                            
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
        
        #sorts all the found values for w from low to high and chooses the lowest one
        norms = sorted([n for n in opt_dict])
        opt_choice = opt_dict[norms[0]] #Optimal choice of 'w' and 'b'
        self.w = opt_choice[0]
        self.b = opt_choice[1]
        
        latest_optimum = opt_choice[0][0]+step*2
        
    
    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b) #Gets the sign '+' or '-' from this equation so it can classify a new point
        if classification !=0 and self.visualization: #If the datafeature is not on the hyperplane, this will run.
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()

#Making a dictionary with some data to use
data_dict = {-1:np.array([[1,7],[2,8],[3,8],]), 
             1:np.array([[5,1],[6,-1],[7,3],])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)
svm.visualize()


