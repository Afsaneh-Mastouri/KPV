#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 12:05:57 2021

@author: afsaneh
"""
import csv

import pickle
import numpy as np
import pandas as pd 
import os
import glob


results_df=pd.DataFrame()
for f in glob.glob('*.p'):
   # print (f)   
        dict1=pickle.load(open( f, "rb" ) )
        df=dict1[1000].iloc[:]
        df_index =df['do_A'].copy()
        results_df=pd.concat([results_df,df],axis=1)
        
        
results_index=df_index#results_df.loc[:,~results_df.columns.duplicated()]
results_df=results_df.drop(['do_A'],axis=1)
mean=results_df.mean(axis=1)
std=results_df.var(axis=1)
m_ps=mean+std
m_ns=mean-std
results_summary=pd.concat([results_index,mean,std,m_ps,m_ns],axis=1)
results_summary.columns=['do_A','Ew_Haw','std','ub','lb' ]
  

do_cal = np.load('/Users/afsaneh/Documents/UCL/Thesis/KPV/AG/Final/Final/do_A_seed1009_std.npz')          
do_A=do_cal['do_A']
gt_EY_do_A=do_cal['gt_EY_do_A']




                
fig=plt.figure(2)
ax=fig.gca() 
plt.scatter(do_A,gt_EY_do_A, c='orange', marker="o", label='y(do(A)')
plt.scatter(results_summary['do_A'],results_summary['Ew_Haw'], c='gray', marker="o", label='Ave_w(H)')
plt.scatter(results_summary['do_A'],results_summary['ub'], c='gray')
plt.scatter(results_summary['do_A'],results_summary['lb'], c='gray')
##plt.scatter(do_A,gt_EY_do_A)
##plt.scatter(Ewh[0],Ewh[1])
ax.set_xlabel('A', fontweight ='bold') 
plt.title("size= {size}".format(size=1000))
##plt.title('MSE={a:.4f}'.format(a=Mse))
plt.legend(loc='upper right');
##plt.scatter(do_A,gt_EY_do_A)
plt.savefig('uniform_normal_Size'+str(1000)+'.png')
plt.show()
plt.close()
  
    
    
    
