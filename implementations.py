    # -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


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
    
    #input_data_tx = np.c_[np.ones(len(input_data)), input_data]  #Adding column of ones to make it tX
    
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

def var_clean_columns(x, var_threshold):
    """
    Deletes columns based on variance threshold, deletes columns with variance under var_threshold
    Arguments:  x (array to be cleaned)
                var_threshold (threshold of variance)
    """
    
    nb_col = len(x[0])
    col_to_del = []
    x_cop = x
    
    for col in range(0,nb_col):
        var = np.var(x_cop[:,col])
        if var <= var_threshold:
            col_to_del.append(col)
            
    x_cop =  np.delete(x_cop, col_to_del, 1)
    
    return x_cop

def clean_undef_columns(x,undef_val):
    
    nb_col = len(x[0])
    col_to_del = []
    x_cop = x
    
    for col in range(0,nb_col):
        val = x_cop[1,col]
        
        if val == undef_val:
            col_to_del.append(col)   
            
    x_cop =  np.delete(x_cop, col_to_del, 1)
    
    return x_cop

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


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


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

            
def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function returns the matrix formed
    # by applying the polynomial basis to the input data
    if degree==0:
        return np.ones(len(x))
    
    tx = np.c_[np.ones(len(x)), x]
    for j in range(2,degree+1):
        tx = np.c_[tx,x**j ]
    return tx

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

def cross_validation_new(weights,tX_cross,y_cross,values):
    """Calculate the cross validation for the 4 way split"""
    y_pred_cross = np.dot(tX_cross, weights[0])
    
    for i in range(1,4):
        curr_indices=np.where(tX_cross==values[i])[0]
        y_pred_cross[curr_indices]= np.dot(tX_cross[curr_indices],weights[i])   
        
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



"""
************************************************************************************************
************************************************************************************************
************************************************************************************************
************************************************************************************************
************************************************************************************************
"""


"""Function which implement various machine learning algorithms for project 1."""

def compute_loss(y, tx, w):
    """Calculate the loss"""
    e = y-(tx@w)
    N = y.size
    L = np.sum(np.square(e))/(2*N)
    
    return L

#**************************************************    GRADIENT DESCENT   *************************************************************

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y-(tx@w)
    N = len(e)
    return -tx.T@e/N
 

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        #compute gradient and loss
        grad=compute_gradient(y,tx,w)
        loss =compute_loss(y,tx,w)
        
        #Update step
        w = w-gamma*grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws
#******************************************************************************************************************************************





#**************************************************    STOCHASTIC GRADIENT DESCENT   *************************************************************

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e=  y- np.dot(tx,w)
    return (-1.0/len(y))*np.dot( np.transpose(tx),e)



def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""  
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter=0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        n_iter+=1
        #compute gradient and loss
        grad=compute_gradient(minibatch_y,minibatch_tx,w)
        loss =compute_loss(minibatch_y,minibatch_tx,w)        
        w = w-gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))   
       

    return losses, ws
#******************************************************************************************************************************************






#**************************************************    LEAST SQUARES   *************************************************************
def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T@tx
    b = tx.T@y
    wts = np.linalg.solve(a,b)
    mse = compute_loss(y, tx, wts)
    return mse, wts    # returns mse, and optimal weights   
#******************************************************************************************************************************************






#**************************************************    RIDGE REGRESSION   *************************************************************
def ridge_regression(y, tx, lambda_):
    """Ridge regression Algorithm"""
    a = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]) + tx.T.dot(tx)
    b = tx.T.dot(y)
    wts = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, wts) + lambda_* np.dot(wts, wts)
    return mse, wts    # returns mse, and optimal weights

#******************************************************************************************************************************************

    
    
    
    
    
#**************************************************    LOGISTIC REGRESSION   *************************************************************
def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_logistic_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    return -np.sum( y*np.log(sigmoid(tx@w)) +  (1-y)*np.log(1-sigmoid(tx@w))  ) 


def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)

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

#**********************************************************************************************************************************************





#**************************************************   REGULARIZED LOGISTIC REGRESSION   *************************************************************


def reg_logistic_gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_):
    """
    Computes the regularized logistic gradient descent
    """
    # init parameters
    threshold = 1e-8
    
    ws = [initial_w]
    losses = []
    w=initial_w
    
    for iter in range(max_iters):
        # get loss and update w 
        loss= calculate_logistic_loss(y, tx, w) + lambda_* (np.sum( np.dot(w.T, w)))
        grad= calculate_logistic_gradient(y, tx, w) +  2*lambda_*w
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

#*******************************************************************************************************************************************************




#**************************************************   NEWTON'S METHOD LOGISTIC REGRESSION   *************************************************************

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    S=np.identity(len(y))
    for i in range(0,len(y)):
        S[i,i] = (sigmoid(tx@w)*(1-sigmoid(tx@w)))[i,0]
    hess = tx.T @ S @ tx
    return hess


def newt_logistic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
    Computes the Newton Logistic Gradient Descent
    """
    # init parameters
    threshold = 1e-8
    
    ws = [initial_w]
    losses = []
    w=initial_w
    
    for iter in range(max_iters):
        # get loss and update w 
        loss= calculate_logistic_loss(y, tx, w) 
        grad= calculate_logistic_gradient(y, tx, w) 
        hess= calculate_hessian(y, tx, w)
        w= w- gamma* ( np.linalg.solve(hess, np.identity(len(w)) )  ) @grad 
        ws.append(w)
        losses.append(loss)
        
        # log info
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            

    return losses, ws

def newt_stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size):
    # init parameters
    threshold = 1e-8
    
    ws = [initial_w]
    losses = []
    w=initial_w
    n_iter=0
    for baty, batx in batch_iter(y, tx, batch_size,max_iters):
            
            loss= calculate_logistic_loss(baty, batx, w) 
            grad= calculate_logistic_gradient(baty, batx, w) 
            hess= calculate_hessian(baty, batx, w)
            w= w- gamma* ( np.linalg.solve(hess, np.identity(len(w)) )  ) @grad 
            ws.append(w)
            losses.append(loss)

            # log info
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
            n_iter+=1
            # converge criterion
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break


    return losses, ws
#*******************************************************************************************************************************************************