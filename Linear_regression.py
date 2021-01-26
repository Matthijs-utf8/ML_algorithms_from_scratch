# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:10:51 2019

@author: matth
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random 

style.use('fivethirtyeight')

#Arrays with the nessescary points you want to calculate the best fit line of
#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

#Creating a dataset to test the algorithm, hm is how many data point you want
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1 #Starting value
    ys = [] #Empty list of ys
    for i in range(hm): #Create a value for how long the range should be
        y = val + random.randrange(-variance, variance) #Pick a random y-value between the negative of the varance and the positive variance
        ys.append(y) #Append the y value to the list of ys
        if correlation and correlation == 'pos':
            val += step #For positive correlation, add the stepsize to the starting value
        elif correlation and correlation == 'neg':
            val -= step #For negative correlation, subtract stepsize from the y value generated
    xs = [i for i in range(len(ys) ) ]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

#Calculating the slope of the best fit line
def best_fit_slope_and_intercept(xs, ys):
    m = ( ( (mean(xs) * mean(ys) ) - mean(xs * ys) ) / ( (mean(xs) * mean(xs) ) - mean(xs * xs) ) )
    
    b = mean(ys) - m * mean(xs)
    return m, b

#Calculating the squared error of every individual data point to the regression line
def squared_error(y_point, y_regr_line):
    return sum( (y_regr_line - y_point) ** 2)

#Calculation the coefficient of determination
def coefficient_of_determination(y_point, y_regr_line):
    y_mean_line = [mean(y_point) for y in y_point]
    squared_error_regr = squared_error(y_point, y_regr_line)
    squared_error_y_mean = squared_error(y_point, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

#Actually assigning values to the defined functions and getting return values
xs, ys = create_dataset(40, 40, 2, correlation='pos')
m, b = best_fit_slope_and_intercept(xs, ys)

#Makes a coordinate of the regressionline for every value of x in xs
regression_line = [(m*x)+b for x in xs]

#Making a prediction at x = 8
predict_x = 50
predict_y = (m * predict_x + b)

#Calculating r_squared
r_squared = coefficient_of_determination(ys, regression_line)
print('Coefficient of determination = ' + str(r_squared))

#Plot the data
plt.scatter(xs, ys) #plot the individual points
plt.scatter(predict_x, predict_y)
plt.plot(xs, regression_line) #plots xs against the regression point
plt.show()


