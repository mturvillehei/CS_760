#~~~~ Morgan Turville-Heitz ~~~~#
#~~~~ 10/8/2023 ~~~~#
#~~~~ CS 760 Fall 2023 ~~~~#

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import warnings
from sklearn.metrics import auc

### Plotting the data as requested for part 1
def part_1(data):

    ### Generating an evenly spaced mesh grid
    grid = np.linspace(-2, 2, 41)
    X, Y = np.meshgrid(grid, grid)

    plt.figure(figsize=(6, 6))
    plt.plot(X, Y, marker='.', color='k', linestyle='none')

    ## Drawing the grid
    for x_val in grid:
        plt.axvline(x=x_val, color='k', linestyle='none', linewidth=0.5)
    for y_val in grid:
        plt.axhline(y=y_val, color='k', linestyle='none', linewidth=0.5)

    ## Drawing the data
    for index, row in data.iterrows():

        x, y, label = row['x1'], row ['x2'], row['y']
        if label == 0:
            plt.plot(x,y,marker='+',color='b',linestyle='none',markersize=8)
        else:
            plt.plot(x, y, marker='o', color='r', linestyle='none', markersize=8, 
                     markerfacecolor='none', markeredgecolor='r')

    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.title('Plot of X1, X2, with data overlain')
    ### Coloring the legend to match the data 
    legend_elements = [Line2D([0], [0], marker='.', color='k', label='Grid',
                              markersize=10, linestyle='none'),
                       Line2D([0], [0], marker='+', color='b', label='y = 0',
                              markersize=8, linestyle='none',),
                       Line2D([0], [0], marker='o', color='r', label='y = 1',
                              markersize=8, linestyle='none', 
                              markerfacecolor='none', markeredgecolor='r')]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.grid(False)  
    plt.show()


## For reference, unused
def EuclideanDistance(row1, row2):
    d2 = np.sum(row1 - row2)**2
    d = np.sqrt(d2)
    return d

### Part 2 Problem 2
def kNN_1(data):

    split_size = 1000

    label = 'Prediction'

    ### Distance metric - Euclidean
    
    for i in range(0, 5):
        print(f"Iteration {i+1}")
        
        split_indices = [(split_size*i), split_size+(split_size*i)-1]
        #print(split_indices)

        test = data.loc[split_indices[0]:split_indices[1]].copy()
        train = data.loc[~data.index.isin(test.index)]
        ### I was experiencing exceedingly long computation times, so I vectorized the entire process.
        for index, row1 in test.iterrows():

            ### Vectorized the operation to help with computation time
            di = np.sqrt(np.sum((train.iloc[:, :-1].values - row1.values[:-1])**2, axis=1))
            ### Nearest neighbor will have the index of the lowest element in di.
            nearest = np.argmin(di)
            test.at[index, 'NewPrediction'] = train.iloc[nearest][label]
        
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        ### Manually counting each
        for index, row in test.iterrows():
            if (row['NewPrediction'] == 1) & (row[label] == 1):
                TP += 1
            elif (row['NewPrediction'] == 1) & (row[label] == 0):
                FP += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 1):
                FN += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 0):
                TN += 1
        accuracy = (TP + TN) / (TP + TN + FN + FP)

        precision = TP / (TP + FP)

        recall = TP / (TP + FN)

        ### I just copied my terminal output into the final results table for the assignment
        print(f"For split indices {split_indices}, Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}.")


# Part 2 problem 3
# b, theta1, theta2
# 1, x1, x2
## Using batch for the group calculations of part 3
## In a batch calculation, the matrix math is  X Theta, rather than Theta xi, and I am operating on the entire X matrix.
def sigmoidBatch(theta, X):
    
    with warnings.catch_warnings():

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        ### [3001 x 4000 dot 3001 x 1]
        tx = np.dot(X, theta)
        #print(tx.shape) #### [4000 x 1]
        s = 1 / (1 + np.exp(-tx))
        #print(s.shape) #### [4000 x 1]
        return s
    
## Using a different function for the predictions
## Here, for individual predictions, I use thetaT Xi
def sigmoid(theta, X):
    with warnings.catch_warnings():

        warnings.filterwarnings('ignore', category=RuntimeWarning)
        tx = np.dot(theta.T, X)
        s = 1 / (1 + np.exp(-tx))
        #print(s)
        return s

### Part 2 Problem 3
def logisticRegression(data, epochs, learning_rate, c):
    results = []
    split_size = 1000

    label = 'Prediction'

    backup = data.copy()

    for i in range(0, 5):
        data = backup
        print(f"Iteration {i+1}")
        split_indices = [(split_size*i), split_size+(split_size*i)-1]

        test = data.loc[split_indices[0]:split_indices[1]].copy()
        train = data.loc[~data.index.isin(test.index)]

        #### Regression
        ### Creating a 0th entry for each column, including the x0 = 1 column.
        # I've already added the bias term before this point.
        theta = np.zeros(train.shape[1]-1)

        for j in range(0, epochs):

            ### Batch calculation to reduce computation time (remarkably faster, in fact)
            XT = train.drop(columns=[label]).values #### 4000 rows x 3001 columns [b + 3000]
            yt = train[label].values
            s = sigmoidBatch(theta, XT)
            err = s - yt #### 4000 x 1

            ### Gradient update
            gradient = np.matmul(XT.T, err) / (yt.size) #### 3001 x 1 
            theta -= learning_rate * gradient  
            
        for index, row in test.iterrows():
            result = sigmoid(theta.transpose(), row.drop(label).values)
            if result > c:
                test.at[index, 'NewPrediction'] = 1
            else:
                test.at[index, 'NewPrediction'] = 0

        ####
        
        TP = 0
        TN = 0
        FN = 0
        FP = 0

        ### Again, counting manually
        for index, row in test.iterrows():
            if (row['NewPrediction'] == 1) & (row[label] == 1):
                TP += 1
            elif (row['NewPrediction'] == 1) & (row[label] == 0):
                FP += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 1):
                FN += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 0):
                TN += 1
        ### I was running into issues initially with TP + FN = 0.
        try:
            accuracy = (TP + TN) / (TP + TN + FN + FP)
        except:
            accuracy = 0
        try:    
            precision = TP / (TP + FP)
        except:
            precision = 0
        
        try:    
            recall = TP / (TP + FN)
        except:
            recall = 0 

        ### Manually copying my terminal output into the latex table.
        print(f"With c = {c}, for split indices {split_indices}, Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}.")

        results.append([accuracy, precision, recall, TP, TN, FP, FN])


    return results


### Plotting for part 4 - accuracy vs k
def acc_k(result_out):

    N_values = [result[0] for result in result_out]
    accuracies = [result[1] for result in result_out]

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('kNN 5-Fold Cross Validation')
    plt.xlabel('k')
    plt.ylabel('Average accuracy')
    plt.grid(True)
    plt.xticks(N_values)  
    
    # Display the plot
    plt.show()

### Part 2 Problem 4
### N is the number of neighbors used in the calculation
def kNN_n(data,N):
    
    acc = 0
    prec = 0
    rec = 0

    split_size = 1000

    label = 'Prediction'

    ### Distance metric - Euclidean
    
    for i in range(0, 5):
        print(f"Iteration {i+1}")
        split_indices = [(split_size*i), split_size+(split_size*i)-1]
        #print(split_indices)

        test = data.loc[split_indices[0]:split_indices[1]].copy()
        train = data.loc[~data.index.isin(test.index)]
        #print(train.head)
        for index, row1 in test.iterrows():
            su = 0
            ### Vectorized the operation 
            di = np.sqrt(np.sum((train.iloc[:, :-1].values - row1.values[:-1])**2, axis=1))
            
            ### Updated to simply sum the nearest neighbor prediction
            nearest = np.argsort(di)[:N]
            for k in nearest:
                su += train.iloc[k][label]
            avg = su/N

            ### Picking the more common. Since N = 10 is allowed, I'm assuming random for equal P(0), P(1)
            if avg > 0.5:
                test.at[index, 'NewPrediction'] = 1
            elif avg < 0.5:
                test.at[index, 'NewPrediction'] = 0  
            elif avg == 0.5:
                test.at[index, 'NewPrediction'] = np.random.choice([0, 1])

        TP = 0
        TN = 0
        FN = 0
        FP = 0
        ### Counting manually
        
        for index, row in test.iterrows():
            if (row['NewPrediction'] == 1) & (row[label] == 1):
                TP += 1
            elif (row['NewPrediction'] == 1) & (row[label] == 0):
                FP += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 1):
                FN += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 0):
                TN += 1
        accuracy = (TP + TN) / (TP + TN + FN + FP)

        precision = TP / (TP + FP)

        recall = TP / (TP + FN)
        ###Results for each k
        print(f"For split indices {split_indices}, Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}.")
        acc += accuracy
        prec += precision
        rec += recall
    ### Averaging for the returned results
    acc /= 5
    prec /= 5
    rec /= 5
    return [N, acc, prec, rec]

### Part 5
### Almost identical to the kNN_N function, except I set the test data range to [2000:2999]
### and vary the classification threshold, i.e. number of neighbors needed for a positive classification.
def part_5_kNN(data):
    kNN = []
    ROC_kNN = [0, 1, 2, 3, 4, 5]
    i = 2

    split_size = 1000

    label = 'Prediction'

    ### Distance metric - Euclidean
    for l in ROC_kNN:
        split_indices = [(split_size*i), split_size+(split_size*i)-1]

        test = data.loc[split_indices[0]:split_indices[1]].copy()
        train = data.loc[~data.index.isin(test.index)]

        for index, row1 in test.iterrows():
            su = 0
            ### Vectorized the operation 
            di = np.sqrt(np.sum((train.iloc[:, :-1].values - row1.values[:-1])**2, axis=1))
            
            ### Updated to simply sum the nearest neighbor prediction
            nearest = np.argsort(di)[:5]
            for k in nearest:
                su += train.iloc[k][label]
            #Technically both should be normalized to the probability (l / N, su / N), but it changes nothing
            if su >= l:
                test.at[index, 'NewPrediction'] = 1
            else:
                test.at[index, 'NewPrediction'] = 0                 


        TP = 0
        TN = 0
        FN = 0
        FP = 0
        ### Counting manually
        
        for index, row in test.iterrows():
            if (row['NewPrediction'] == 1) & (row[label] == 1):
                TP += 1
            elif (row['NewPrediction'] == 1) & (row[label] == 0):
                FP += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 1):
                FN += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 0):
                TN += 1
        accuracy = (TP + TN) / (TP + TN + FN + FP)

        precision = TP / (TP + FP)

        recall = TP / (TP + FN)

        FPR = FP / (TN + FP)
        
        print(f"For threshold {l}, Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}.")

        kNN.append([l, recall, accuracy, FPR])

    return kNN

def part_5_LR(data, C):
    epochs = 4
    learning_rate = 0.005
    results = []
    split_size = 1000
    i = 2
    label = 'Prediction'
    data.insert(0, 'Bias', 1)

### Varying my classification threshold.
    for c in C:
        print(f"Iteration {i+1}")
        split_indices = [(split_size*i), split_size+(split_size*i)-1]

        test = data.loc[split_indices[0]:split_indices[1]].copy()
        train = data.loc[~data.index.isin(test.index)]

        #### Regression
        ### Creating a 0th entry for each column, including the x0 = 1 column.
        theta = np.zeros(train.shape[1]-1)

        for j in range(0, epochs):

            ### Batch calculation to reduce computation time (remarkably fast)
            XT = train.drop(columns=[label]).values #### 4000 rows x 3001 columns [b + 3000]
            yt = train[label].values
            s = sigmoidBatch(theta, XT)
            err = s - yt #### 4000 x 1

            ### Gradient update
            gradient = np.matmul(XT.T, err) / (yt.size) #### 3001 x 1 
            theta -= learning_rate * gradient  
            
        for index, row in test.iterrows():
            result = sigmoid(theta.transpose(), row.drop(label).values)
            
            if result > c:
                test.at[index, 'NewPrediction'] = 1
            else:
                test.at[index, 'NewPrediction'] = 0
        
        TP = 0
        TN = 0
        FN = 0
        FP = 0

        for index, row in test.iterrows():
            if (row['NewPrediction'] == 1) & (row[label] == 1):
                TP += 1
            elif (row['NewPrediction'] == 1) & (row[label] == 0):
                FP += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 1):
                FN += 1
            elif (row['NewPrediction'] == 0) & (row[label] == 0):
                TN += 1
        
        print(f"TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}")
        try:
            accuracy = (TP + TN) / (TP + TN + FN + FP)
        except:
            accuracy = 0
        try:    
            precision = TP / (TP + FP)
        except:
            precision = 0
        
        try:    
            recall = TP / (TP + FN)
        except:
            recall = 0 

        print(f"With c = {c}, for split indices {split_indices}, Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}.")
        FPR = FP / (TN + FP)
        results.append([c, recall, accuracy, FPR])

    return results

def plot_roc_curves(log_reg_results, knn_results):

    _, log_reg_recall, _, log_reg_fpr = zip(*log_reg_results)
    _, knn_recall, _, knn_fpr = zip(*knn_results)
    
    log_auc = auc(log_reg_fpr, log_reg_recall)
    knn_auc = auc(knn_fpr, knn_recall)

    plt.figure(figsize=(10, 7))
    plt.plot(log_reg_fpr, log_reg_recall, label=f'Logistic Regression (AUC = {log_auc:.2f})', color='blue')
    plt.plot(knn_fpr, knn_recall, label=f'kNN, k = 5 (AUC = {knn_auc:.2f})', color='green')
    
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()


fn = 'D2z.txt'

D2z = pd.read_csv(fn, delim_whitespace=True, names=['x1', 'x2', 'y'])
#part_1(D2z)
fn2 = 'emails.csv'

emails = pd.read_csv(fn2)

### Sampling ruined my accuracy for some reason.
#emails = emails.sample(frac=1).reset_index(drop=True)

emails.drop(['Email No.'], axis=1,inplace=True)
### For part 2
#kNN_1(emails)

## For part 4
N = [1, 3, 5, 7, 10]
result_out = []
for n in N:
    r = kNN_n(emails, n)
    result_out.append(r)
    print(f"Accuracy = {r[1]} for N = {n}")


kNN_results = part_5_kNN(emails)
C = np.linspace(0,1,21)
LR_results = regression_results = part_5_LR(emails, C)
plot_roc_curves(LR_results, kNN_results)

acc_k(result_out)

### Adding a bias column here before my logistic regression script.
emails.insert(0, 'Bias', 1)
epochs = 1000

nu = 0.005
C = np.linspace(0, 1, 25)
for c in C:
     results = logisticRegression(emails, epochs, nu, c)


