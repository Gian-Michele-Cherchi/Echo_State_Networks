# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 01:44:49 2020

@author: gmche
"""

import numpy as np 
import matplotlib.pyplot as plt
import data_analysis as data
from sklearn.linear_model import Ridge
import signalz
import seaborn as sns 

#Mackey_series= fts.data.mackey_glass.get_data(b=0.1, c=0.2, tau=17, initial_values=np.array([0.5, 0.55882353, 0.61764706, 0.67647059, 0.73529412, 0.79411765, 0.85294118, 0.91176471, 0.97058824, 1.02941176, 1.08823529, 1.14705882, 1.20588235, 1.26470588, 1.32352941, 1.38235294, 1.44117647, 1.5 ]), iterations=1000)

mackey_series_ = signalz.mackey_glass(2500, a=0.2, b=0.8, c=0.9, d=23, e=10, initial=0.4)

#mapping data in range [1,-1]
mackey_series, angcoff, intercept = data.linear_map(-1, 1, mackey_series_)


#HYPERPARAMETERS OPTIMIZATION 
def opt_ens(alpha,radius,reservoir,n_train,n_av):
    leak = 0.9
    ignore_points = 100
    sparsity = 0.35
    rms_error = np.zeros([len(radius), len(alpha)])
    for l in range(len(radius)): 
        for j in range(len(alpha)): 
            
           
            esn = data.esn(leak, alpha[j], 0.5, reservoir,ignore_points,1,sparsity, radius[l])
            rms = np.zeros(n_av)
            for m in range(n_av):
                
                X0 = np.ones(reservoir)
                n_prd = 10
                prediction = []
                for i in range(n_prd):
                    w_out,X0_prd = esn.train(X0, mackey_series[i:n_train+i].reshape(-1,1), mackey_series[i+1:n_train+i+1])
                    y_prd = esn.predict(X0_prd,mackey_series[i+1:n_train+i+1].reshape(-1,1),w_out)
                    y_prd_scaleback = (y_prd[len(y_prd)-1]-intercept)/angcoff
                    prediction.append(y_prd_scaleback)
                
                somma= 0 
                
                #computes the quadratic loss between the predicted values and the trained values 
                for k in range(len(prediction)):
                    somma+= (prediction[k] - mackey_series_[n_train:n_train+n_prd][k])**2
                rms[m] =  np.sqrt(somma/len(prediction))
                
            # average quadratic loss over different training prediction session 
            rms_error[l,j] = np.mean(rms)

    return rms_error


spec_radius = [round(x,2) for x in np.linspace(0.9,1.3,30)]
reg_factor =[round(x,2) for x in np.linspace(1,7,30)]


rms_opt_map = opt_ens(reg_factor,spec_radius, 40,2000,10) 


#plots an heatmap for each couple of the hyperparameters 
sns.heatmap(rms_opt_map, xticklabels= reg_factor,yticklabels= spec_radius )
plt.show()



