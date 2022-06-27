#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:33:58 2021

@author: afsaneh
"""
import os,sys,datetime
import time
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple
import numpy as np 
import operator
from scipy.stats import norm

from functools import partial
import matplotlib as mpl
import statistics 
import itertools as it
import pickle

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax.scipy.linalg as jsla

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from functools import partial


#%matplotlib inline
plt.show()

from utils import *



#directory = "/Users/afsaneh/Documents/UCL/Thesis/KPV/"
#for item in os.listdir(directory):
#    if item.endswith(".p"):
#       os.remove(os.path.join(directory, item))
       

#from GenScm import *
 #%Generators parameters

#param
N=[5,100,500,1000,3000,7000,10000]
n=N[-1]
p=N[0]
train_sz=N[5]

do_A_min, do_A_max, n_A = -2, 2, 50


#%Generators parameters

#[u,a,y,z,w]
m_e=[4,0,0,0,0]
m_u=[0,0,0]
v_u=[1,1,1]
m_z=[0,0]
v_z=[3,1]
m_w=[0,0]
v_w=[1,3]




#%Generative models 

def gen_sigma(p=10):
    #s=jnp.eye(p)
    s=[]
    for i in range(p):
        for j in range(p):
            if j==i: s.append(1)
            elif abs(i-j)==1: s.append(0.5)
            else: s.append(0)
    return jnp.array(s).reshape(p,p)

def gen_alpha(p):
    return jnp.array([b**(-2) for b in range(1,p+1)])

    
def gen_U(n,key):      
    e1=random.uniform(key[0],(n,),minval=0,maxval=1)
    U2=3*random.uniform(key[1],(n,),minval=0,maxval=1)-1   
    e3= np.where((U2>1),0,-1)   
    e4= np.where((U2<0),0,-1) 
    e5=(e3+e4)
    U1=e1+e5+1
    return U1,U2


def gen_Z(U1,U2, m_z, v_z, n,  key):
    ##Z1= U1*0.25+ (random.normal(key[0],(n,))*v_z[0])+m_z[0]
    ##Z2= U2**2 + random.uniform(key[1],(n,),minval=0,maxval=1)
    ##Z1= U1+ (random.normal(key[0],(n,))*v_z[0])+m_z[0]
    ##Z2= U2+ (random.normal(key[1],(n,))*v_z[1])+m_z[1]
    Z1= U1+ (random.normal(key[0],(n,))*v_z[0])+m_z[0]
    Z2= U2+ random.uniform(key[1],(n,),minval=-1,maxval=1)
    #Z2= U2+ (random.normal(key[1],(n,))*v_z[1])+m_z[1]
    return Z1,Z2
    #return Z2


def gen_W(U1,U2, m_w, v_w, n,  key):
    ##W1= U1+ random.normal(key[0],(n,))*v_w[0]+m_w[0]
    ##W2= U2+ random.normal(key[1],(n,))*v_w[1]+m_w[1]
    W1= U1+ random.uniform(key[0],(n,),minval=-1,maxval=1)
    #W1= U1+ (random.normal(key[0],(n,)) *v_w[0])+m_w[0]
    W2= U2+ (random.normal(key[1],(n,)) *v_w[1])+m_w[1]
    return W1,W2
    #return W1


def gen_X(p ,n, key):
    if p!=0:
        m_x=np.zeros(p)
        sigma=gen_sigma(p)
        X=np.random.multivariate_normal(m_x, sigma, n)
    else: X=jnp.asarray([])
    return X.T

def gen_A(U1,U2,beta, n, key):    
    A= U2 + random.normal(key,(n,)) * beta    
    return A


def gen_Y(A, U1, U2, n):
    y= U2*(np.cos(2*(A+.3*U1+.2)))
    return y



#% generative dist

sigma = gen_sigma(p)
alpha =  gen_alpha(p)

seed_list={}
sd_lst= [5949, 7422, 4388, 2807, 5654, 5518, 1816, 1102, 9886, 1656, 4379,
       2029, 8455, 4987, 4259, 2533, 9783, 7987, 1009, 2297] #np.random.choice(10000,20)

for sd in sd_lst[:1]:
            
            seed=[]
            seed1=sd+5446
            seed2=sd+3569
            seed3=sd+10
            seed4=sd+1572
            seed5=sd+42980
            seed6=sd+368641
            seed=[seed1,seed2,seed3,seed4,seed5,seed6]
            num_var=10
            seed_list[sd]=seed
            
            #% generate random state
            #extra 2keys for choosing train_test data subsets
            
            keyu = random.PRNGKey(seed1)
            keyu, *subkeysu = random.split(keyu, 4)
            
            keyz = random.PRNGKey(seed2)
            keyz, *subkeysz = random.split(keyz, 4)
            
            
            keyw = random.PRNGKey(seed3)
            keyw, *subkeysw = random.split(keyw, 4)
            
            
            keyx = random.PRNGKey(seed4)
            keyx, *subkeysx = random.split(keyx, 100)
            
            keya = random.PRNGKey(seed5)
            keya, *subkeysa = random.split(keya, 100)

         # Un_Standardised sample
              
            U = ((jnp.asarray(gen_U(n,key=subkeysu))).T)                                 
            Z = ((jnp.asarray(gen_Z(U[:,0],U[:,1], m_z, v_z, n, key=subkeysz))).T) 
            W = ((jnp.asarray(gen_W(U[:,0],U[:,1], m_w, v_w, n, key=subkeysw))).T) 
            X=  (gen_X(p,n, keyx).T)
            A = (gen_A(U[:,0],U[:,1],0.05, n, key=keya))
            Y = (gen_Y(A, U[:,0], U[:,1], n))
            
            # Standardised sample
            Us = standardise(U) [0]
            Zs = standardise(Z) [0]
            Ws = standardise(W) [0]
            Xs = standardise(X) [0]
            As = standardise(A) [0]
            Ys = standardise(Y) [0]
            
            
            fig=plt.figure(2)
            ax=fig.gca()             
            plt.scatter(As,Ys, c='orange', marker="o", label='ys ~ As')
            plt.scatter(A,Y, c='blue', marker="o", label='y ~ A')
            ax.set_xlabel('A', fontweight ='bold') 
            plt.legend(loc='upper left')
            plt.title('Standar vs non_standardised data')
            plt.show()
            
            '''non Standardised sample'
            U = (jnp.asarray(gen_U(n,key=subkeysu))).T                              
            Z = (jnp.asarray(gen_Z(U[:,0],U[:,1], m_z, v_z, n, key=subkeysz))).T)[0] 
            W = standardise(gen_W(U[:,1], m_v,n, key=subkeysv[1])) [0]
            X=  standardise(gen_X(p,n, keyx).T) [0]
            A = (gen_A(U[:,0],U[:,1],0.05, n, key=keya)) 
            '''
            n_eval=10000
            seed_eval=49765
            key_ev= random.PRNGKey(seed_eval)
            key_ev, *subkeys_ev = random.split(key_ev, 4)
            
            
            # causal ground truth standardised
            do_A_scaled = jnp.linspace(do_A_min, do_A_max, n_A)
            do_A = standardise(A)[1].inverse_transform(do_A_scaled)
            Uns=(jnp.asarray(gen_U(n=n_eval,key=subkeys_ev))).T
            Uss=standardise(Uns)[0]
            EY_do_A_s=standardise(jnp.array([(gen_Y(A=a, U1=Uns[:,0], U2=Uns[:,1], n=n_eval)).mean() for a in do_A]))[0]
            EY_do_A=jnp.array([(gen_Y(A=a, U1=Uns[:,0], U2=Uns[:,1], n=n_eval)).mean() for a in do_A])
            
            '''
            plt.scatter(do_A_scaled,EY_do_A_s), 
            plt.title('GT standardised'), plt.show()
            plt.scatter(do_A,EY_do_A), 
            plt.title('GT Non_standardised'), plt.show()
            '''
            
            
            fig=plt.figure(2)
            ax=fig.gca()             
            plt.scatter(do_A_scaled,EY_do_A_s, c='orange', marker="o", label='ys ~ do(As)')
            plt.scatter(do_A,EY_do_A, c='blue', marker="o", label='y ~ do(A)')
            ax.set_xlabel('A', fontweight ='bold') 
            plt.legend(loc='upper right')
            plt.title('Standar vs non_standardised data')
            plt.show()

 
                        
            ''' 
            np.savez('do_A_seed{}_std.npz'.format(sd),
                     do_A = do_A_scaled,
                     gt_EY_do_A = EY_do_A_s)
            '''
            
            np.savez('do_A_seed{}_std.npz'.format(sd),
                     do_A = do_A,
                     gt_EY_do_A = EY_do_A)
        
        
            '''
            train_u, test_u = Us[:train_sz], Us[train_sz:]
            train_z, test_z = Zs[:train_sz], Zs[train_sz:]
            train_w, test_w = Ws[:train_sz], Ws[train_sz:]
            train_x, test_x = Xs[:train_sz], Xs[train_sz:]
            train_a, test_a = As[:train_sz], As[train_sz:]
            train_y, test_y = Ys[:train_sz], Ys[train_sz:]
            '''
            
            train_u, test_u = U[:train_sz], U[train_sz:]
            train_z, test_z = Z[:train_sz], Z[train_sz:]
            train_w, test_w = W[:train_sz], W[train_sz:]
            train_x, test_x = X[:train_sz], X[train_sz:]
            train_a, test_a = A[:train_sz], A[train_sz:]
            train_y, test_y = Y[:train_sz], Y[train_sz:]
           
            ''      
            np.savez('main_seed{}_std.npz'.format(sd),
                     splits=['train', 'test'],
                     train_y=train_y,
                     train_a=train_a,
                     train_z=train_z,
                     train_w=train_w,
                     train_u=train_u,
                     train_x=train_x,
                     test_y = test_y,
                     test_a = test_a,
                     test_z = test_z,
                     test_w = test_w,
                     test_u = test_u
                     ,test_x = test_x
                     )



np.savez('seed_lst.npz', seed_list)
