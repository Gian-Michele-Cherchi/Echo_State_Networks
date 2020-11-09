# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:01:35 2020

@author: gmche
"""
import numpy as np 
from sklearn.linear_model import Ridge
import scipy.sparse as sp 
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt 




class esn:
    def __init__(self, leak_rate,lambda_ ,a, input_size,points_ignore,n_features,sparsity, spectral_radius):
        
        #PARAMETERS: using private attributes 
        self.__stocks = n_features
        self.__a = a
        self.__leak = leak_rate
        self.__rad = spectral_radius
        self.__nx = input_size
        self.__sparsity = sparsity 
        self.__lambda = lambda_
        self.__m = points_ignore
        
        #initialize reservoir matrix and input matrix: probably the same for training and prediction
        #input matrix 
        self.__w_in= np.random.uniform(-self.__a,self.__a ,  (self.__nx,self.__stocks+1))
        #self.__w_in[:,0] = np.random.uniform(-0.1,0.1,size = self.__nx)
        self.__w_in[:,0] = np.zeros(self.__nx)
        
        #reservoir matrix and setting of spectral radius
        x = sp.rand(self.__nx, self.__nx, density= self.__sparsity )
        self.__w_0 = (2*x-np.ones([self.__nx,self.__nx]))/2
        self.__eig = eigs(self.__w_0)
        self.__radius = max([abs(self.__eig[0][i]) for i in range(len(self.__eig[0]))])
        self.__w = (self.__rad*self.__w_0)/self.__radius
        
        
    def train(self, X0_tr, u_tr, y_tr):
        
        #deisgn matrix
        self.__u_tr = u_tr
        self.__y_tr = y_tr
        self.__npoints = len(self.__u_tr) 
        self.__X = np.zeros([self.__npoints+1, self.__nx])
        self.__X[0][:] = X0_tr
        #z-matrix 
        self.__z = np.zeros([self.__npoints,self.__nx + self.__stocks + 1])
        
        self.__u_tr_1 = np.zeros([self.__npoints,self.__stocks+1])
        for i in range(self.__npoints):
            self.__u_tr_1[i][0] = 1
            self.__u_tr_1[i][1:] = self.__u_tr[i][:]
       
       
        for i in range(1,self.__npoints+1):
            #self.__b = self.__w_in*self.__u_tr_1[i-1][:]
            #print(self.__b.shape)
            self.__X[i][:] = (1-self.__leak)*self.__X[i-1][:] + self.__leak*np.tanh(self.__w.dot(self.__X[i-1][:]) + self.__w_in.dot(self.__u_tr_1[i-1][:]))
            #build concatenation matrix z 
            self.__z[i-1][0] = 1
            self.__z[i-1][1:self.__stocks] = self.__u_tr[i-1][:]
            self.__z[i-1][self.__stocks+1:] = self.__X[i][:]
                    
        self.__S = self.__z[self.__m+1:][:]
        
        if self.__stocks == 1: self.__D = self.__y_tr[self.__m+1:]
        else: self.__D = self.__y_tr[self.__m+1:][:]
        
        #print(self.__S.shape, self.__D.shape)
        self.__regression = Ridge(self.__lambda)
        self.__fits = self.__regression.fit(self.__S,self.__D)
        self.__w_out = self.__fits.coef_
        
        #plt.plot(self.__X[self.__npoints][:])
        return self.__w_out, self.__X[self.__npoints][:]
    
    
    
    def predict(self,X0_prd ,u_test,w_out):
        self.__w_out = w_out
        #design matrix
        self.__X = np.zeros([self.__npoints, self.__nx])
        self.__X[0][:] = X0_prd
        self.__u_test = u_test
        self.__npoints = len(self.__u_test)
        #z-matrix 
        self.__z = np.zeros([self.__npoints,self.__nx + self.__stocks + 1])
        
        self.__u_test_1 = np.zeros([self.__npoints,self.__stocks+1])
        for i in range(self.__npoints):
            self.__u_test_1[i][0] = 1
            self.__u_test_1[i][1:] = self.__u_test[i][:]
        #print(self.__u_test_1.shape)
        
        self.__X[1][:] = (1-self.__leak)*self.__X[0][:] + self.__leak*np.tanh(self.__w.dot(self.__X[0][:]) + self.__w_in.dot(self.__u_test_1[0][:]))
        self.__z[0][0] = 1
        self.__z[0][1:self.__stocks] = self.__u_test[0][:]
        self.__z[0][self.__stocks+1:] = self.__X[0][:]
        
        for i in range(2,self.__npoints):
            #update design matrix at time i 
            self.__X[i][:] = (1-self.__leak)*self.__X[i-1][:] + self.__leak*np.tanh(self.__w.dot(self.__X[i-1][:]) + self.__w_in.dot(self.__u_test_1[i][:]))
            #build concatenation matrix z 
            self.__z[i][0] = 1
            self.__z[i][1:self.__stocks] = self.__u_test[i][:]
            self.__z[i][self.__stocks+1:] = self.__X[i][:]
        #print(self.__w_out.shape, self.__z.shape)
        self.__y_prd = self.__w_out.dot(self.__z.T)
        
        return self.__y_prd



# Class that reads the .csv file and analize data by columns (by field) or by rows 
class get_csv: 
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            header = f.readline().strip().split(',')
            print(header)
            self.__columns = {field : [] for field  in header}
            for row in f: 
                row_elements = row.strip().split(',')
                for row_element,field in zip(row_elements, self.__columns.keys()):
                    try: 
                        self.__columns[field].append(float(row_element))
                    except:
                        self.__columns[field].append(row_element)
            for key in header: 
                try:
                    self.__columns[key].remove('')
                except:
                    pass
                
    def get_field_column(self, field_name):
        return self.__columns[field_name]
    def get_row(self, row_index):
        riga = [self.__columns[x][row_index] for x in self.__columns.keys()]
        return riga
##############################linearly map 
##############################time series data vector in the range [minimum, maximum]
def linear_map(minimum, maximum, v):
     try:
         size = len(v)
         min_vector = np.array([min(v[i]) for i in range(size)])
         max_vector = np.array([max(v[i]) for i in range(size)])
         ang_coeff = 2/(max_vector-min_vector)
         intercept = (minimum*max_vector- maximum*min_vector)/(max_vector-min_vector)
         v = [ang_coeff[i]*v[i] + intercept[i] for i in range(size)]
         
     except:
             min_vector = min(v)
             max_vector = max(v)
             ang_coeff = 2/(max_vector-min_vector)
             intercept = (minimum*max_vector- maximum*min_vector)/(max_vector-min_vector)
             v = ang_coeff*v + intercept
     return v, ang_coeff, intercept
 ###############################################construct the design matrix from the time series vector
def design_matrix(v,n_features,n_points):
    des_mat = np.zeros([n_points,n_features])
    for i in range(n_points):
        des_mat[i][:] = v[i:i+n_features]
    return des_mat 
    
 


    
    
    
    