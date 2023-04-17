# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:13:30 2023

@author: Wei
"""

import numpy as np
from scipy.special import expit
from tqdm.auto import tqdm
from sklearn.metrics import log_loss

tqdmSwitch = True

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

def Logistic_Regression_Batch_GD(x, y, eta, K, q=None):
    #Initialize weights and bias
    b = 0.1
    w = np.zeros([x.shape[1],1])
    n = len(x)
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Make prediction
        y_pred = bound(sigmoid(x@w+b))
        
        #Update weights
        w = w - eta*(x.T@(y_pred-y)/n)
        b = b - eta*np.sum(y_pred-y)
        
        #Compute cost
        costs += [log_loss(y, y_pred)]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Logistic_Regression_SGD(x, y, eta, K, q=None):
    #Initialize weights and bias
    b = 0.1
    w = np.zeros([x.shape[1],1])
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(x))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        #a = eta/np.sqrt(k+1)
        
        #Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        #Update weights
        grad = (xx*(y_pred-yy)).reshape((64,1))
        w = w - a*grad
        b = b - a*(y_pred-yy)

        #Compute cost
        pred = bound(sigmoid(x@w+b))
        costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Logistic_Regression_GA(x, y, eta, K, q=None):
    #Initialize weights and bias
    b = 0.1
    w = np.zeros([x.shape[1],1])
    g = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(x))
        xx = x[idx]
        yy = y[idx]
        
        #Fixed learning rate
        a = eta
        #a = eta/np.sqrt(k+1)
        
        #Make prediction
        y_pred = bound(sigmoid(xx@w+b))

        #Update weights
        n = k+1
        grad = (xx*(y_pred-yy)).reshape((64,1))
        g += grad
 
        w = w - a*(g/n)
        b = b - a*(y_pred-yy)

        #Compute cost
        pred = bound(sigmoid(x@w+b))
        costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Logistic_Regression_SAG(x, y, eta, K, q=None):
    #Initialize weights and bias
    b = 0.1
    w = np.zeros([x.shape[1],1])
    g = np.zeros((len(x), 64,1)) #Gradient table
    g_prev = 0 #Previous gradient
    idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(x))
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
        grad = (xx*(y_pred-yy)).reshape((64,1))
 
        #Update weights
        w = w - a*(grad - g_prev + np.sum(g, axis=0))/m
        b = b - a*(y_pred-yy)
        
        g[idx] = grad
        g_prev = grad

        #Compute cost
        pred = bound(sigmoid(x@w+b))
        costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Logistic_Regression_SAG_L2(x, y, eta, K, L, q=None):
    #Initialize weights and bias
    b = 0.1
    w = np.zeros([x.shape[1],1])
    g = np.zeros((len(x), 64,1)) #Gradient table
    g_prev = 0 #Previous gradient
    idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(x))
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
        grad = (xx*(y_pred-yy)).reshape((64,1))
 
        #Update weights
        w = w - a*((grad/m - g_prev/m + np.sum(g, axis=0)/m) + L*w)
        b = b - a*(y_pred-yy)
        
        g[idx] = grad
        g_prev = grad

        #Compute cost
        pred = bound(sigmoid(x@w+b))
        costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)

def Logistic_Regression_SAGA(x, y, eta, K, q=None):
    #Initialize weights and bias
    b = 0.1
    w = np.zeros([x.shape[1],1])
    g = np.zeros((len(x), 64,1)) #Gradient table
    g_prev = 0 #Previous gradient
    idxs = []
    m = 0
    
    costs = []
    y = y.reshape((len(y),1))
    
    #For each iteration
    for k in tqdm(range(K), disable=tqdmSwitch):
        
        #Draw random sample with replacement
        idx = np.random.randint(0,len(x))
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
        grad = (xx*(y_pred-yy)).reshape((64,1))
 
        #Update weights
        w = w - a*(grad - g_prev + np.sum(g, axis=0)/m)
        b = b - a*(y_pred-yy)
        
        g[idx] = grad
        g_prev = grad

        #Compute cost
        pred = bound(sigmoid(x@w+b))
        costs += [log_loss(y, pred, labels = [0,1])]
        
    if q != None:
        q.put([w, b, np.array(costs)])
        
    return w, b, np.array(costs)