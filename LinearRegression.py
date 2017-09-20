#importing the libraries and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston

#load the data into the variable
boston = load_boston()

#load the data & target into a DataFrame object
df = pd.DataFrame(boston.data, columns= boston.feature_names)
target = pd.DataFrame(boston.target, columns= ['TGT'])

#concatenate the data and target into a single DataFrame object ({axis=0 : rows, axis=1 : columns })
df = pd.concat([df, target], axis=1)

#Split the overall data into 80% of Taining data and 20% of Test data
datadiv = int(0.8 * len(df))

x_train = df['RM'][:datadiv]
y_train = df['TGT'][:datadiv]

x_test = df['RM'][datadiv:]
y_test = df['TGT'][datadiv:]

#hypothesis function  h(x) = mx + b
def h(m,x,b):
    return (m*x+b)

#cost function   J(x) = (1/(2*N))*[(y - h(x))]^2
def calculate_error(target,m,b,x):
    N=len(x)
    error = ((target-h(m,x,b))**2).sum()
    return (1/N)*error

#calculate the gradient at one point , gradient_m and gradient_b is the partial differentail of J(x) wrt m and b respectively
def step_gradient(m, b, x, target, learningRate):
    N = len(x)
    gradient_m = ((-2 / N) * x * (target - h(m,x,b))).sum()
    gradient_b = ((-2 / N) * (target - h(m,x,b))).sum()
    new_m = m - (gradient_m * learningRate)
    new_b = b - (gradient_b * learningRate)
    return [new_m, new_b]

#perform gradient descent of data using step_gradient function 
def perform_gradient_descent(initial_m, initial_b, x, target, learningRate):
    m, b = initial_m, initial_b
    i = 1
    while True:
        e1 = calculate_error(target,m,b,x)
        m1,b1 = m,b 
        m, b = step_gradient(m, b, x, target, learningRate)
        if i % 1000 == 0:
            print("i:{}, m:{}, b:{}, error:{}".format(i, m, b,calculate_error(target,m,b,x)))
        e2 = calculate_error(target,m,b,x)
        i += 1
        if e2 > e1:
            break
    return m1, b1


#initailizing the values
initial_m = 9.4
initial_b = -35.7
learningRate = 0.01

#print the initial error, perform gradient descent, iterate for multiple iterations for which the the error is least by changing the initial vaules of m and b 
print("Starting Gradient Descent with m={0}, b = {1} & error={2}".format(initial_m, initial_b, calculate_error(target=y_train, m=initial_m, b=initial_b, x=x_train)))

final_m, final_b = perform_gradient_descent(initial_m, initial_b, x_train,
                                            y_train, learningRate)

print("Final values after Gradient Descent are m={0}, b = {1} & error={2}".format(final_m, final_b, calculate_error(target=y_train,
                                                                                                                    m=final_m,
                                                                                                                    b=final_b,
                                                                                                                    x=x_train)))
#plot the training data and target which includes an intercept which best represents the data
plt.plot(x_train,y_train,'r.')
plt.plot(x_train, h(final_m, x_train, final_b), 'b')
plt.show()