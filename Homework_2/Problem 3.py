#~~~~ Morgan Turville-Heitz ~~~~#
#~~~~ 09/24/2023 ~~~~#
#~~~~ CS 760 Fall 2023 ~~~~#

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

### Reading data
fn = 'Dbig.txt'

data = pd.read_csv(fn, delim_whitespace=True, names=['x1', 'x2', 'y'])

### Using the same subsets as in problem 2.7
Dsub = [32, 128, 512, 2048, 8192]

### Pulling test data out after shuffling
data_ = data.sample(frac=1).reset_index(drop=True)
test = data_.iloc[8193:]

output = []

for sub in Dsub:
    ### Grabbing the subset
    D = data_.iloc[:sub]

    #Using unlimited number of nodes because the dataset is small and I'm curious how well scikit is able to classify compared to my algorithm
    Y = D['y']
    X = D[['x1', 'x2']]

    #Training. Arbitrarily set random state to 0.
    root = DecisionTreeClassifier(random_state=0, )
    root.fit(X,Y)
    ### Number of nodes
    N = root.tree_.node_count

    Xtest = test[['x1', 'x2']]
    Ytest = test['y'] 

    ### MSE = MAE, again. Technically using MAE, but it's boolean, so the values are identical.
    pred = root.predict(Xtest)
    err = 0
    for i, p in enumerate(pred):
        print(f"prediction is {p}")
        print(f"target is {Ytest.iloc[i]}")
        if p != Ytest.iloc[i]:
            err += 1
    err /= len(Ytest)

    output.append([sub, N, err])


print(output)

n_values = [row[0] for row in output]
err_n_values = [row[2] for row in output]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, err_n_values, marker='o')
plt.xscale('log')  # Setting x-axis to logarithmic scale as n values are powers of 2

for n, err_n in zip(n_values, err_n_values):
    plt.annotate(f'({n}, {err_n:.4f})', (n, err_n), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('n')
plt.ylabel('err_n')
plt.title('Learning curve for training set of size n')
plt.grid(True)
plt.show()