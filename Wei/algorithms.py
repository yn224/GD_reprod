# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:13:30 2023

@author: Wei
"""

import numpy as np
from scipy.special import expit
from tqdm.auto import tqdm
from sklearn.metrics import log_loss, mean_squared_error

tqdmSwitch = False
weightEvalRes = 5000

def convert_labels(y):
    y = np.where(y<5, 0, y)
    return np.where(y>4, 1, y)

def sigmoid(x):
    return expit(x)

def bound(x):
    x = np.where(x<1e-16,1e-16,x)
    return np.where(x>1-(1e-16), (1-1e-16), x)

def Predict(x,w,b):
    pred = bound(sigmoid(x@w+b))
    pred = np.where(pred>0.5, 1, pred)
    return np.where(pred<=0.5, 0 ,pred)

def Linear_Predict(x,w):
    return x@w

#---------- Logistic Regression Algorithms ----------#
def Logistic_Regression_SGD(x, y, eta, K, L=0, q=None):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        #a = eta/np.sqrt(k+1)
        
        #Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        #Update weights
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        w = w - a*(grad + L*w)

        #Compute cost
        if ((k)%weightEvalRes==0):
            pred = bound(sigmoid(x@w+b))
            costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Logistic_Regression_SAG(x, y, eta, K, L=0, q=None):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    g = np.zeros((x.shape[0], x.shape[1],1)) #Gradient table
    G = np.zeros_like(w) #Gradient table sum
    idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        
        #Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        #Check if data point has been seen
        if idx not in idxs:
            idxs += [idx]
            m += 1
        
        #Calculate current gradient
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        #Update gradient table
        G = G - g[idx] + grad
        #Update previous sample gradient
        g[idx] = grad
        #Update weights
        w = w - a*(G/m + L*w)

        #Compute cost
        if ((k)%weightEvalRes==0):
            pred = bound(sigmoid(x@w+b))
            costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Logistic_Regression_SAGA(x, y, eta, K, L=0, q=None):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    g = np.zeros((x.shape[0], x.shape[1],1)) #Gradient table
    G = np.zeros_like(w) #Gradient table sum
    idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        
        #Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        #Check if data point has been seen
        if idx not in idxs:
            idxs += [idx]
            m += 1
        
        #Calculate current gradient
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        #Update weights
        w = w - a*((grad - g[idx] + G/m) + L*w)
        #Update gradient table
        G = G - g[idx] + grad
        #Update previous sample gradient
        g[idx] = grad

        #Compute cost
        if ((k)%weightEvalRes==0):
            pred = bound(sigmoid(x@w+b))
            costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Logistic_Regression_finito(x, y, mu, K, q=None):
    # Initialize weights and bias
    # b = 0.1
    w = np.zeros([x.shape[1],1])
    P = np.zeros((x.shape[0], x.shape[1], 1)) # Weight table
    G = np.zeros((x.shape[0], x.shape[1], 1)) # Gradient table
    grad = np.zeros_like(w)
    idxs = []
    m = 0
    b = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    # TODO: you can change this to whatever alpha value it is
    # - pg. 7 of Finito paper.
    alpha = 1

    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):        
        # Learning rate - mu is convexity constant
        #a = 1/(alpha * mu * (k+1))
        a = mu

        # Draw random sample with replacement
        idx = np.random.randint(0,len(x))
        xx = x[idx]
        yy = y[idx]

        # Check if data point has been seen
        if idx not in idxs:
            idxs += [idx]
            m += 1
        
        # Take the average of the weights \phi
        phi_bar = np.sum(P, axis=0) / m

        # Take the average of the gradients
        g_bar = np.sum(G, axis=0) / m

        # Update weight
        w = phi_bar - a * g_bar
        #b = b - a * (y_pred-yy)
        
        # Store data
        P[idx] = w
        G[idx] = grad
        
        # Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        # Calculate current gradient
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))

        #Compute cost
        if ((k)%weightEvalRes==0):
            pred = bound(sigmoid(x@w+b))
            costs += [log_loss(y, pred, labels = [0,1])]

    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)


#---------- Linear Regression Algorithms ----------#
def Linear_Regression_SGD(x, y, eta, K, L=0, q=None):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        #a = eta/np.sqrt(k+1)
        
        #Make prediction
        y_pred = xx@w

        #Update weights
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        w = w - a*(grad + L*w)

        #Compute cost
        if ((k+1)%weightEvalRes==0):
            pred = x@w
            costs += [mean_squared_error(y, pred, squared=False)/x.shape[0]]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Linear_Regression_SAG(x, y, eta, K, L=0, q=None):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    g = np.zeros((x.shape[0], x.shape[1],1)) #Gradient table
    G = np.zeros_like(w) #Gradient table sum
    idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        
        #Make prediction
        y_pred = xx@w

        #Check if data point has been seen
        if idx not in idxs:
            idxs += [idx]
            m += 1
        
        #Calculate current gradient
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        #Update gradient table
        G = G - g[idx] + grad
        #Update previous sample gradient
        g[idx] = grad
        #Update weights
        w = w - a*(G/m + L*w)

        #Compute cost
        if ((k+1)%weightEvalRes==0):
            pred = x@w
            costs += [mean_squared_error(y, pred, squared=False)/x.shape[0]]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Linear_Regression_SAGA(x, y, eta, K, L=0, q=None):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    g = np.zeros((x.shape[0], x.shape[1],1)) #Gradient table
    G = np.zeros_like(w) #Gradient table sum
    idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        
        #Make prediction
        y_pred = xx@w

        #Check if data point has been seen
        if idx not in idxs:
            idxs += [idx]
            m += 1
        
         #Calculate current gradient
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        #Update weights
        w = w - a*((grad - g[idx] + G/m) + L*w)
        #Update gradient table
        G = G - g[idx] + grad
        #Update previous sample gradient
        g[idx] = grad

        #Compute cost
        if ((k+1)%weightEvalRes==0):
            pred = x@w
            costs += [mean_squared_error(y, pred, squared=False)/x.shape[0]]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

#---------- Others ----------#
def Logistic_Regression_Batch_GD(x, y, eta, K, q=None):
    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    n = x.shape[0]
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Make prediction
        y_pred = bound(sigmoid(x@w+b))
        
        #Update weights
        w = w - eta*(x.T@(y_pred-y)/n)
        #b = b - eta*np.sum(y_pred-y)
        
        #Compute cost
        costs += [log_loss(y, y_pred)]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)
def Logistic_Regression_GA(x, y, eta, K, q=None):

    #Initialize weights and bias
    b = 0
    w = np.zeros([x.shape[1],1])
    g = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(y))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        #a = eta/np.sqrt(k+1)
        
        #Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        #Update weights
        n = k+1
        grad = (xx*(y_pred-yy)).reshape((x.shape[1],1))
        g += grad
 
        w = w - a*(g/n)
        #b = b - a*(y_pred-yy)

        #Compute cost
        pred = bound(sigmoid(x@w+b))
        costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)