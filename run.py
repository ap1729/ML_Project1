import numpy as np
import matplotlib.pyplot as plt

#********************************************************FUNTIONS********************************************************
    # -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv


def standardize(x):
    """
    Standardize the original data set with standard deviation.
    Arguments: x (array to be standardized)
    """
    std_x = np.std(x)
    if std_x == 0:
        std_x = 1
    mean_x = np.mean(x)
    x = x - mean_x
    x[np.where(x==0)[0]]=1e-5
    x = x / std_x
    return x

def quart_standardize(x):
    """
    Standardize the original data set with quantiles
    Arguments: x (array to be standardized)
    """
    q1 = np.percentile(x,15)
    q3 = np.percentile(x,85)
    iqr = q3-q1
    if iqr == 0:
        iqr = 1
    median = np.median(x)
    x = x-median
    x[np.where(x==0)[0]]=1e-5
    x = x/iqr
    return x

def scaled_standardize(x):
    """
    Standardize the original data set with max values
    Arguments: x (array to be standardized)
    """
    min_x = np.min(x)
    max_x = np.max(x)
    range_x = max_x-min_x
    if range_x == 0:
        range_x = 1
    x = x - min_x
    x[np.where(x==0)[0]]=1e-5
    x = x/range_x
    return x

def standardize_data(input_data, mode = 0):
    """
    Standardize the original data with respect to the mode chose
    Arguments: input_data (array to be standardized)
                mode = 0, 1, 2 (0: standard deviation
                                1: quantile standardization
                                2: scaled standardization)
    """
    #Standardize all the remaining columns
    for i in range(0,len(input_data[0])):
        if mode == 0:
            input_data[:,i] = standardize(input_data[:,i])
        elif mode == 1:
            input_data[:,i] = quart_standardize(input_data[:,i])
        elif mode == 2:
            input_data[:,i] = scaled_standardize(input_data[:,i])
        else:
            input_data[:,i] = standardize(input_data[:,i])
    
    return input_data

def clean_columns_rows(x, y, percentage_lign, percentage_col, flags):
    """
    Deletes columns and rows if number of -999. values is higher than a certain percentage of number of columns or rows
    Arguments:  x (array to be cleaned)
                y (prediction to be cleaned [to respect the size of matrices])
                percentage_lign (value between 0 and 1, multiplies the number of ligns for comparison)
                percentage_col (value between 0 and 1, multiplies the number of columns for comparison)
                flags (function flags for feature expansion)
    """
    nb_col = len(x[0])
    nb_lign = len(x)
    col_to_del = []
    lign_to_del = []
    x_cop = x
    y_cop = y
    
    for col in range(0,nb_col):
        unique, counts = np.unique(x_cop[:,col], return_counts = True)
        
        if flags[0] == True and counts[np.where(unique == 0.)[0]] > 0.8*nb_lign:
            col_to_del.append(col)
            
        if counts[np.where(unique == -999.)[0]] > percentage_lign*nb_lign:
            col_to_del.append(col)
            
    x_cop =  np.delete(x_cop, col_to_del, 1)
    
    for lign in range(0,nb_lign):
        unique, counts = np.unique(x[lign,:], return_counts = True)
        
        if counts[np.where(unique == -999.)[0]] >= percentage_col*nb_lign:
            lign_to_del.append(lign)
            
    x_cop = np.delete(x_cop, lign_to_del, 0)
    y_cop = np.delete(y_cop, lign_to_del, 0)
    
    return x_cop, y_cop, col_to_del

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:,2:]
    
    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def predict_labels_new(weights, data,degree,flags, col_to_del):
    """Generates class predictions given weights, and a test data matrix"""
    #After Splitting into 4 parts, we need to recombine them here
    y_pred=np.zeros(data.shape[0])
    
    data23=[row[22] for row in data[0:100]] #23rd column which is JET_PRI_NUM and has exactly 4 distinct values
    values=(list(set(data23))) #get all 4 unique values , and these have been verified to correspond to our original ordering
    
    first_col = np.array(data[:,0])
    first_col[first_col == -999.] = 0
    first_col[first_col == 0.] = np.median(first_col)
    data[:,0] = first_col
    
    for i in range(0,4):
        #extracting the indices of column 22
        curr_indices=np.where(data==values[i])[0]
        
        data_cop = data[curr_indices]
        data_cop = np.delete(data_cop, col_to_del[i], 1)
        
        X_test = standardize_data(data_cop, mode = 1)
        data_cop[data_cop == 0.]=1e-3
        
        tX_pred=feature_expand(data_cop,degree,flags[i]) #Feature Expand and delete 23rd column
        y_pred[curr_indices]= np.dot(tX_pred,weights[i])   
        
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed and shuffle both x and y with the same seed
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    train_rows = int(ratio * len(x))
    test_rows = len(x) - train_rows
    x_train=x[0:train_rows]
    y_train=y[0:train_rows]
    x_test=x[train_rows:]
    y_test=y[train_rows:]
    return x_train, x_test, y_train, y_test

def cross_validation(weights,tX_cross,y_cross):
    """Calculate the cross validation"""
    y_pred_cross = np.dot(tX_cross, weights)
    
    y_pred_cross[np.where(y_pred_cross <= 0)] = -1
    y_pred_cross[np.where(y_pred_cross > 0)] = 1  
    #calculate fraction which is correct with the known data
    return (np.count_nonzero(y_pred_cross == y_cross))/len(y_cross)

def feature_expand(x,degree,flags=[True,True,True]):
    """Form (tx) to get feature expanded data in matrix form."""
        
    tx = np.c_[np.ones(len(x)), x] # Adding column of ones
    
    for j in range(2,degree+1):
        tx = np.c_[tx, x**j ]
        
    tx = np.c_[tx, np.sqrt(np.abs(x)) ]
    
    if flags[0]==True:
        tx=np.c_[tx,np.log(np.abs(x))]
    
    if flags[1]==True:
        tx=np.c_[tx,np.sin(x)]
        
    if flags[2]==True :
        tx = np.c_[tx,np.cos(x)]
        
    return tx

def logistic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """compute the logistic gradient descent"""
    # init parameters
    threshold = 1e-8
    
    ws = [initial_w]
    losses = []
    w=initial_w
    
    for iter in range(max_iters):
        # get loss and update w 
        loss= calculate_logistic_loss(y, tx, w)
        grad= calculate_logistic_gradient(y, tx, w)
        w = w-gamma*grad
        ws.append(w)
        
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            

    return losses, ws

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_logistic_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    return -np.sum( y*np.log(sigmoid(tx@w)) +  (1-y)*np.log(1-sigmoid(tx@w))  ) 

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)
#************************************************************************************************************************


DATA_TRAIN_PATH = 'train.csv' 
y_orig, X_raw, ids = load_csv_data(DATA_TRAIN_PATH) #Standardized data ready to use

#NOTE: ONLY RUN FIRST TWO CELLS BEFORE THIS, ONLY LOAD DATA
# DONT RUN FEATURE EXPANSION OR CROSS VALIDATION, DOING IT HERE
tX23=[row[22] for row in X_raw[0:100]] #23rd column which is JET_PRI_NUM and has exactly 4 distinct values
values=(list(set(tX23))) #get all 4 unique values
indices=[]

weights_new = []
loss_new   = []

#Split each of the 4 subsets of Training Data into 2 parts one- for training and other for cross-validating with a fixed seed

#Cross Validation Parameters
ratio=0.97 #percentage of data to be trained and what to be used for cros-validating
seed=13 #random seed

#Feature Expansion Parameters
degree=1 #degree atleast 1

#Logistic Regression Parameters
gamma=[0.000002,0.0000055,0.0000054,0.0000056]
flags = [[1,1,1],[0,1,1],[0,1,1],[0,1,1] ]
max_iters = 1200 

cross_val= [0,0,0,0] #cross validation score for each group
shapes=[]

# Replace every -999. in the first column by the median of the said column to avoid having it deleted

first_col = np.array(X_raw[:,0])
first_col[first_col == -999.] = 0
first_col[first_col == 0.] = np.median(first_col)
X_raw[:,0] = first_col

col_to_del = []

#X_reduced, y_reduced = clean_columns_rows(X_raw, y_orig, 0, 1, [1])
#tX_reduced = standardize_data(X_reduced, mode = 0)

for i in range(0,4): #iterate over each of the 4 possibilites of pri num
    curr_indices = []
    curr_indices=np.where( [row[22] for row in X_raw] ==values[i])[0] 
    indices.append(curr_indices) #keeps track of indices

    X_reduced, y_reduced, del_col = clean_columns_rows(X_raw[curr_indices,:], y_orig[curr_indices], 0, 1, flags[i])

    #Standardize all the remaining columns
    X_reduced = standardize_data(X_reduced, mode = 1)

    #tX_curr=np.delete(tX_raw[curr_indices], 22, 1) #take current indices and delete 23rd col

    tX_4way_temp = feature_expand(X_reduced,degree,flags[i]) #Feature Expansion
    print(tX_4way_temp.shape)
    y_4way_temp = y_reduced

    #Cross validation split    
    tX_4way_train,tX_4way_cross,y_4way_train,y_4way_cross = split_data(tX_4way_temp,y_4way_temp,ratio,seed)

    shapes.append(y_4way_train.shape[0])

    #Running Logistic Regression ***********************

    # Initialization
    w_initial = np.zeros(len(tX_4way_train[0]))
    y_4way_train[np.where(y_4way_train==-1)]=0 # Since y is array with -1 and +1 , we need to make it to 0's and 1's


    # Start Logistic gradient descent
    logistic_losses, logistic_ws = logistic_gradient_descent(y_4way_train, tX_4way_train, w_initial, max_iters, gamma[i])

    weights_new.append(logistic_ws[-1])
    loss_new.append(logistic_losses[-1])

    cross_val[i] = cross_validation(logistic_ws[-1],tX_4way_cross,y_4way_cross)

    col_to_del.append(del_col)

shapes=np.array(shapes)

DATA_TEST_PATH = 'test.csv' 
_, X_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = 'output.csv' 

y_pred = predict_labels_new(weights_new, X_test,degree,flags, col_to_del)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)