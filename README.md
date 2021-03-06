# KPV
This depository contains implementation of Kernel Proxy Variable (KPV) approach, the first method suggested in 'Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction' (https://arxiv.org/pdf/2105.04544.pdf). 
The code for Proxy Maximum Moment Restriction (PMMR), the second method proposed in the paper can be find at https://github.com/yuchen-zhu/kernel_proxies.

# How to run the code?
The main accepts the observerd sample in form of dictionaries with seperate label for training (for calculation of stage 1 and stage 2) and test (calculationof causal effect based on causal function estimated using training data). 
To run the code you need to:
1) copy/download main + utils + cal_alpha 
2) Add path/address of True_Caual_Effect.npz to load #do_cal at line #66 of main.py
3) Add path/address of Data_Sample/*.npz to load samples at line #71 of main.py. Training and test sample saved as data disciotnaries.

Results are compressed and saved at the same directory/path as the main.py. 


# Acknowledgments
The initial implementation of KPV based on step by step calculation of causal effect (according to Proposition 2 of paper) was slow in estimating causal effect from large samples n>1000. Thanks to [Liyuan Xu](https://www.ly9988.work), we have improved implementation by replacing the final step of calculating alpha by the function utils.stage2_weights from his repo on DeepFeatureIV (nice trick! [Liyuan](https://www.ly9988.work)). 




