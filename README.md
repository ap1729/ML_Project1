Run on Python 3.8.10 with numpy installed.

The project has two python files - Implementations.py and Submissions.py.

Implementations.py - This file has all functions used for 
a)Loading data from a csv file- using helper functions
b)Pre-processing data - standardizing data in different ways, implementingfeature expansion
c)Cross-Validation- splitting the dataset initially and later to compute the cross-validation value.
d)Algorithms-  GD,SGD,least squares, ridge regression, logistic regression and regularized logistic regression.

Submissions.py- This file has our implementation of our best algorithm- which is splitting the data into 4 parts and running logistic regression on our data which has been pre-processed (standardized, deleted columns, replaced -999 values by the median and feature expansion).

Running: Run Submissions.py with 2 csv files in the same path- "train.csv" with the training data and "test.csv" with the test data. 

Output: "Output.csv" with 2 columns - Ids and the prediction according to our algorithm.
