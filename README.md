# shallowgibbs-doublebackpropagation-notebook
A Notebook to Describe how Double Backpropagation is applied to the Shallow Gibbs Model or any alike-Structured Model

```python
import numpy as np
import numpy.ma as ma
import shallowgibbs.doublebackpropagation as SGDBS
```

To test the model, we have extracted a random Potts Cluster (from a random Potts partition), obtained using the pottscompleteshrinkage package 
(https://pypi.org/project/pottscompleteshrinkage/). 

The design of the Shallow Net is simple as possible:
    
    W = [vec(w1), vec(w2)] with w1 of size (l_0xl_1) and w2 of size (l_1xl_2)
    
In this example, l_1 = 1. 


```python
### Load the Potts Cluster Data

shallowgibbs_covariables_train = np.loadtxt('All_Cluster_xtrain_cluster_1.csv', delimiter=',') 
shallowgibbs_covariables_test = np.loadtxt('All_Cluster_xtest_cluster_1.csv',  delimiter=',')
shallowgibbs_response_train = np.loadtxt('All_Cluster_ytrain_cluster_1.csv',  delimiter=',') 
shallowgibbs_response_test =  np.loadtxt('All_Cluster_ytest_cluster_1.csv', delimiter=',')

### Apply NaNs corrections if there are any.

np.where(np.isnan(shallowgibbs_response_train), ma.array(shallowgibbs_response_train, mask=np.isnan(shallowgibbs_response_train)).mean(axis=0), shallowgibbs_response_train) 
np.where(np.isnan(shallowgibbs_response_test), ma.array(shallowgibbs_response_test, mask=np.isnan(shallowgibbs_response_test)).mean(axis=0), shallowgibbs_response_test) 	


### Load the first initial Predictions from any other model with same structure, or from the Shallow Gibbs Model.

W = np.loadtxt('All_W_train_predictions_cluster_1.csv', delimiter=',')
b = np.loadtxt('All_b_train_predictions_cluster_1.csv',  delimiter=',')
Sigma = np.loadtxt('All_Sigma_train_predictions_cluster_1.csv',  delimiter=',') 

shallowgibbs_Y_testpred =np.loadtxt('All_Cluster_ytest_prediction_cluster_1.csv', delimiter=',')
shallowgibbs_Y_trainpred =np.loadtxt('All_Cluster_ytrain_prediction_cluster_1.csv', delimiter=',')

### Set the parameters

l_0 = shallowgibbs_covariables_train.shape[1]
l_1 = 1
l_2 = shallowgibbs_response_train.shape[1]

### Set the number of DBS iterations

double_backpropagation_time = 3

### Set the dbs learning rate for the parameters: dbs_epsilon

dbs_epsilon = 10e-3

Final_Cluster_ytrain_prediction, Final_SPNNR_train_rmse, Final_Cluster_ytest_prediction, test_rmse = SGDBS.Double_Backpropagated_predictions (shallowgibbs_Y_trainpred, shallowgibbs_Y_testpred, W,b,Sigma, double_backpropagation_time, l_1, shallowgibbs_covariables_train,shallowgibbs_response_train,shallowgibbs_covariables_test,shallowgibbs_response_test,dbs_epsilon)

```

