#~~~~ Morgan Turville-Heitz ~~~~#
#~~~~ 09/21/2023 ~~~~#
#~~~~ CS 760 Fall 2023 ~~~~#

import pandas as pd
import math
import random
import csv
import matplotlib.pyplot as plt
import numpy as np

### Class for the construction of the node.
class DTNode:
    def __init__(self, D):
        #Feature, xj
        self.split_j = None
        #Split c
        self.split_c = None
        #Pointer to left node
        self.split_left = None
        #Pointer to right node
        self.split_right = None
        #Whether this is a leaf node
        self.leaf = None
        #Class of node
        self.leaf_label = None
        #Data subset
        self.data = D

### Determining the best Data split. 
def BestSplit(D, C1, C2):
    #Store c, gain as [c,gain]
    best_1 = [0, -1]
    best_2 = [0, -1]
        
    # Calculating gainratio for each split
    # Storing highest gainratio for both features
    for c in C1:
        
        infogain, gainratio, e_S = Gain(D, 'x1', c)
        if gainratio > best_1[1]:
            best_1 = [c, gainratio]
    for c in C2:
        infogain, gainratio, e_S = Gain(D, 'x2', c)
        if gainratio > best_2[1]:
            best_2 = [c, gainratio]
    
    #Determining the best split by comparing infogain for each features
    if best_1[1] > best_2[1]:
        return ['x1', best_1[0]]
    elif best_1[1] < best_2[1]:
        return ['x2', best_2[0]]
    else:
        #Randomly return either if the peak infogain is identical
        return random.choice([['x1', best_1[0]], ['x2', best_2[0]]])

### Split Entropy. Previously, I calculated this inside Gain, but I was having issues with calling Gain in my DetermineCandidateSplits function, so instead just implemented this.
def split_entropy(D, xj, c):
    D0 = D[D[xj] >= c]
    D1 = D[D[xj] < c]
    
    p0 = len(D0) / len(D)
    p1 = len(D1) / len(D)
    
    # Calculating split entropy
    e_S = 0
    if p0 != 0:
        e_S -= p0 * math.log2(p0)
    if p1 != 0:
        e_S -= p1 * math.log2(p1)

    return e_S


### Calculating infogain, GainRatio, and split information/split entropy
### (This term is a bit confusing, since we refer to multiple values as 'split entropy' between homework and lectures)
def Gain(D, xj, c):

    infogain = 0
    gainratio = 0
    e_dy = 0
    
    #Splitting the data
    D0 = D[D[xj] >= c]
    D1 = D[D[xj] < c]
    
    ### Calculating base entropy of the data
    
    p_1 = len(D[D['y'] == 1]) / len(D['y'])
    p_0 = len(D[D['y'] == 0]) / len(D['y'])

    if p_1 > 0:
        e_dy -= p_1 * math.log2(p_1)
        
    if p_0 > 0:
        e_dy -= p_0 * math.log2(p_0)

    if e_dy == 0:
        print(D)
        print('Zero entropy of the data D (this shouldn\'t be reachable)')
        return infogain, gainratio, e_dy
    
    ### Calculating the conditional entropy e_dyx
    ### If the data is not empty,
    if len(D1['y']) != 0:
        p_d1_1 = len(D1[D1['y'] ==1] ) / len(D1['y'])
        p_d1_0 = len(D1[D1['y'] ==0] ) / len(D1['y'])

    else:
        p_d1_1 = 0
        p_d1_0 = 0
    
    ### and the probabilities both not 0,
    if p_d1_1 != 0 and p_d1_0 != 0:
        
        p_d1 = (len(D1) / len(D)) * (- (p_d1_1 * math.log2(p_d1_1)) - (p_d1_0 * math.log2(p_d1_0)))

    ### Setting conditional probabilities to 0 for P(Y=y|X=x)==0
    else:

        if p_d1_1 == 0 and p_d1_0 != 0:
            p_d1 = (len(D1) / len(D)) * (- (p_d1_0 * math.log2(p_d1_0)))
        elif p_d1_1 != 0 and p_d1_0 == 0:
            p_d1 = (len(D1) / len(D)) * (- (p_d1_1 * math.log2(p_d1_1)))
        else:
            p_d1 = 0

    ### If the data is not empty,
    if len(D0['y']) != 0:

        p_d0_0 = len(D0[D0['y'] ==0] ) / len(D0['y'])
        p_d0_1 = len(D0[D0['y'] ==1] ) / len(D0['y'])
    else:
        p_d0_0 = 0
        p_d0_1 = 0 
    ### and the probabilities both not 0,
    if p_d0_0 !=0 and p_d0_1 !=0:
        ### Then calculate conditional probability.
        p_d0 = (len(D0)/len(D)) * (-(p_d0_1 * math.log2(p_d0_1))-(p_d0_0 * math.log2(p_d0_0)))
    else:
    ### Setting conditional probabilities to 0 for P(Y=y|X=x)==0
        if p_d0_0 == 0 and p_d0_1 != 0:
            p_d0 = (len(D0)/len(D)) * (-(p_d0_1 * math.log2(p_d0_1)))
        elif p_d0_0 != 0 and p_d0_1 == 0:
            p_d0 = (len(D0)/len(D)) * (-(p_d0_0 * math.log2(p_d0_0)))
        else:
            #print('Both have p_d0 == 0')
            p_d0 = 0

    e_dyx = p_d0 + p_d1

    #Infogain = base entropy - conditional entropy following the split
    infogain = e_dy - e_dyx
    
    #Calculating split information/split entropy
    p0 = len(D0) / len(D)
    p1 = len(D1) / len(D)
    
    ### Handling is probabilities are 0 for e_S.
    if p0 !=0 and p1 != 0:
        e_S = -(p1 * math.log2(p1)) - (p0 * math.log2(p0))
    elif p0 !=0 and p1 ==0:
        e_S = -(p0 * math.log2(p0))
    elif p0 ==0 and p1 !=0:
        e_S = -(p1 * math.log2(p1))
    
    ### If both probabilities are 0, let e_S = 0. This is mathematically sound, as e_S = P * log2(P) -> 0 as P -> 0. 
    else:
        e_S = 0



    # If split entropy is 0, return
    if e_S == 0:
        print('Zero entropy of the split')
        print(f"InfoGain is {infogain}")
        return infogain, gainratio, e_S
    
    # Gainratio calculating
    gainratio = infogain / e_S

    return infogain, gainratio, e_S
    

def checkStoppingCriteria(D, C, xj):

    #If Data is empty (realistically we shouldn't reach this, since the above node would be a leaf?)
    if len(D['y']) == 0:

        return True
    
    # Additionally:
    # Primary: If all the labels are in the same class (0 ∨ 1) then stop, i.e. nunique == 1
    if D['y'].nunique() == 1:

        return True

    #If all candidate splits in C have 0 entropy, then all c in C will be removed during the candidate split selection, then C = []. 
    if len(C) == 0:

        return True
    
    # Return False if any gainratio != 0
    # If we reach this point, the only stopping criteria that could be satisfied is all gainratio == 0
    all_GR = all(Gain(D, xj, c)[1] == 0 for c in C)

    return all_GR

def MakeSubtree(D, Halt_At_Root):
    
    ### Initializing the node
    node = DTNode(D)

    #Determining possible splits
    C1, C2 = DetermineCandidateSplits(D, Halt_At_Root)

    #Checking stopping criteria
    stop1 = checkStoppingCriteria(D, C1, 'x1')
    stop2 = checkStoppingCriteria(D, C2, 'x2')

    # For problem 2 part 3:
    if Halt_At_Root:
        #Here, I iterate through each value c for each candidate split. This is so I can access the InfoGain for each candidate.
        #There's a more elegant way to implement this obviously. 
        candidates_1 = []
        candidates_2 = []
        for c in C1:

            info, gain, e = Gain(D, 'x1', c)
            candidates_1.append([c, info, gain, e])

        for c in C2:
            info, gain, e = Gain(D, 'x2', c)
            candidates_2.append([c, info, gain, e])
        filename = 'candidates_output.csv'

        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write the headers
            csvwriter.writerow(['Feature', 'c', 'InfoGain', 'GainRatio', 'SplitEntropy'])

            # Write the data for feature x1
            for row in candidates_1:
                csvwriter.writerow(['x1'] + row)

            # Write the data for feature x2
            for row in candidates_2:
                csvwriter.writerow(['x2'] + row)

        return node

    #If both stopping criteria are met, i.e. for both features we have gainratio == 0 or no candidate splits with nonzero entropy, make the node a leaf
    if stop1 and stop2:
        #Make this node a leaf 
        node.leaf = True
        majorities = D['y'].mode()
        #If multiple majorities, that means N(0) == N(1)
        if len(majorities) > 1:
            #Setting label to 1 if equal number in a leaf
            node.leaf_label = 1
            return node
        else:
            try:
                #Setting the label to the majority value
                node.leaf_label = majorities[0]
                return node
            except:
                #If there is no majority, then the data is empty. Shouldn't happen
                print('How did you mess this up?')
                
    else:
        ### Determining the best split from the candidates
        best = BestSplit(D, C1, C2)
        ### Sets the feature xj and the split c
        node.split_j = best[0]
        node.split_c = best[1]

        ### Creates the next nodes

        #Left for >= , i.e. 'Then'
        D_l = D[D[best[0]] >= best[1]]
        #Right for <, i.e. 'Else'
        D_r = D[D[best[0]] < best[1]]

        #Recursive subtree generation
        node.split_left = MakeSubtree(D_l,Halt_At_Root=False)
        node.split_right = MakeSubtree(D_r,Halt_At_Root=False)

        #Return the node
        return node

def DetermineCandidateSplits(D, Halt_At_Root):
    # Storing values of c for x1, x2
    c1 = []
    c2 = []

    # Sorting values ascending
    x1_sort = D.sort_values(by='x1', ascending=True)
    x2_sort = D.sort_values(by='x2', ascending=True)

    # For each xi, checks if x(i+1) is the same. If not, sets this as a candidate
    for j in range(0, len(x1_sort) - 1):
        if x1_sort['x1'].iloc[j] != x1_sort['x1'].iloc[j + 1]:
            c1.append(x1_sort['x1'].iloc[j])
        if x2_sort['x2'].iloc[j] != x2_sort['x2'].iloc[j + 1]:
            c2.append(x2_sort['x2'].iloc[j])

    # Adds the last candidate, because in the previous expression, if xi_j == xi_j+1, xi_j is not added.
    c1.append(x1_sort['x1'].iloc[-1])
    c2.append(x2_sort['x2'].iloc[-1])

    # Checking split entropy !=0
    to_r = []
    for c in c1:
        e_S = split_entropy(D, 'x1', c)
        if e_S == 0:
            to_r.append(c)

    ### For problem 2 part 3 - I don't want to remove e_S == 0  cases, but instead view their infogain. 
    if not Halt_At_Root:
        c1 = [c for c in c1 if c not in to_r]

    to_r = []
    for c in c2:
        e_S = split_entropy(D, 'x2', c) 
        if e_S == 0:
            to_r.append(c)
    
    ### For problem 2 part 3 - I don't want to remove e_S == 0  cases, but instead view their infogain. 
    if not Halt_At_Root:
        c2 = [c for c in c2 if c not in to_r]

    # If c1 or c2 are empty, this is caught in the CheckStoppingCriteria function
    return c1, c2


def Predict(node, data):

    #Gives a prediction if the node is a leaf.
    if node.leaf:
        return node.leaf_label
    
    #Otherwise, recursively iterates through the function until reaching a leaf.
    else:
        label = node.split_j
        c = node.split_c
        if data[label] >= c:
            left = node.split_left
            return Predict(left, data)
        else:
            right = node.split_right
            return Predict(right,data)

### Chat GPT (GPT-4), accessed on 9/24/2023 at 6:10pm, helped me write the draw_tree function.
def draw_tree(node, x=0.5, y=1, level=1, width=0.7, title='Decision Tree'):

    ### Iterates through each node. If the node is not empty, create a new shape. If the node is a leaf, let it be green; otherwise, blue.
    if node is not None:
        if node.leaf:
            label = f'y = {node.leaf_label}'
            color = 'lightgreen'
        else:
            label = f'{node.split_j} >= {node.split_c}?'
            color = 'lightblue'
        

        plt.gca().add_patch(plt.Circle((x, y), 0.05, fill=True, color=color))
        plt.text(x, y, label, ha='center', va='center')
        
        ### Then, if not a leaf, add a line pointing to the next subset of leafs. Iterate through the draw_tree function down each of these branches until the last leaf is reached.
    
        if not node.leaf:
            left_x = x - width / (1.5 ** level)
            right_x = x + width / (1.5 ** level)
            
            plt.plot([x, left_x], [y - 0.05, y - 0.15], 'k-')
            plt.plot([x, right_x], [y - 0.05, y - 0.15], 'k-')
            
            plt.text((x + left_x) / 2, y - 0.1, 'True', ha='center', va='center')
            plt.text((x + right_x) / 2, y - 0.1, 'False', ha='center', va='center')
            
            ### Iterating down the branches, with inputs for the position of the lower node w.r.t. the upper node.
            draw_tree(node.split_left, left_x, y - 0.15, level + 1, width)
            draw_tree(node.split_right, right_x, y - 0.15, level + 1, width)
    
    # Set the title of the plot
    plt.title(title)

### Looking for the boundaries of the decision space. As mentioned elsewhere, when drawing these boundaries, there shouldn't be overlaps - we partition off a new region of space with each decision.
### There may be overlaps during construction, but once every node is bounded by leaves, the code should only show unique regions.
def getBoundaries(node, path=[]):
    boundaries = []

    if node is not None:
        #Finalize if a leaf
        if node.leaf:
            boundaries.append({
                'path': path,
                'label': node.leaf_label
            })
        ### Search iteratively through the decision tree to find the next set of spaces. 
        else:
            # Internal node, extend the path and recursively search through both trees
            left_path = path + [{'feature': node.split_j, 'value': node.split_c, 'relation': '>='}]
            right_path = path + [{'feature': node.split_j, 'value': node.split_c, 'relation': '<'}]
            ### Then, add spaces for the new nodes. 
            boundaries.extend(getBoundaries(node.split_left, left_path))
            boundaries.extend(getBoundaries(node.split_right, right_path))
    return boundaries

### Includes leaves in the count. 
def nodeCount(node):
    if node is None:
        return 0
    if node.leaf:
        return 1
    else:
        return 1 + nodeCount(node.split_left) + nodeCount(node.split_right)

### Plotting unique boundaries for each bound.
def plotBoundaries(decision_boundaries, title, x1_lims, x2_lims):
    ### Initializing
    plt.figure(figsize=(10, 6))
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)

    plt.xlim(x1_lims[0], x1_lims[1])
    plt.ylim(x2_lims[0], x2_lims[1])

    # Since this is a binary decision tree, we can't have overlapping spaces, i.e. each region is bounded by it's space in the decision tree. There's a proof for this somewhere.
    # Since there isn't an overlap, I can just plot each region separately. 
    for boundary in decision_boundaries:
        path = boundary['path']
        label = boundary['label']
        
        x1_min, x1_max = x1_lims[0], x1_lims[1]
        x2_min, x2_max = x2_lims[0], x2_lims[1]
        
        # Stepping through possible boundary based on the conditions
        for c in path:
            f = c['feature']
            v = c['value']
            r = c['relation']
            ### For 'left'
            if f == 'x1':
                if r== '>=':
                    x1_min = max(x1_min, v)
                else:
                    x1_max = min(x1_max, v)
            ### For 'right'
            elif f == 'x2':
                if r == '>=':
                    x2_min = max(x2_min, v)
                else:
                    x2_max = min(x2_max, v)
        
        #Coloring y = 1 as orange, y = 0 as blue
        color = 'orange' if label == 1 else 'blue'
        plt.fill_betweenx(np.linspace(x2_min, x2_max, 100), x1_min, x1_max, color=color, alpha=0.3)
    #Finally plotting
    plt.legend(handles=[plt.Rectangle((0,0),1,1, color='orange', alpha=0.3), plt.Rectangle((0,0),1,1, color='blue', alpha=0.3)], labels=['Label 1', 'Label 0'])
    plt.show()


##### Rough overview:
##### Load Data
##### Generate Tree:
#####   Determine Split
#####   Check Stopping Conditions for Splits
#####   If Stop:
#####      Generate Node
#####   If Not Stop:
#####      Find Best Split
#####      Create Split
#####      Generate Tree at Split
##### Plot Tree
##### Plot Boundaries
##### Test New Data

#For problem 2.3
Halt_At_Root = False

fn = 'Dbig.txt'
data = pd.read_csv(fn, delim_whitespace=True, names=['x1', 'x2', 'y'])

### Problem 2.7
Dsub = [32, 128, 512, 2048, 8192]

### Permutate the data, then split the data.
data_ = data.sample(frac=1).reset_index(drop=True)
test = data_.iloc[8193:]


output = []


for sub in Dsub:

    #Pulling the first n to form the subset of dat
    D = data_.iloc[:sub]

    #Generating the tree
    root = MakeSubtree(D, Halt_At_Root)

    #Number of nodes
    N = nodeCount(root)
    
    ### Chat GPT (GPT-4), accessed on 9/24/2023 at 6:10pm, helped with writing the draw_tree, plotBoundaries, and getBoundaries function. 
    # Plotting the tree
    plt.figure(figsize=(10, 6))
    draw_tree(root, title = f"Decision tree for N = {sub}")
    plt.axis('off')
    plt.show()

    #Determing the decision boundaries recursively.
    decision_boundaries = getBoundaries(root)

    #title = 'Decision Boundary for ' + str(fn)
    title = 'Decision Boundary for N = ' + str(sub)
    x1_lims = [D['x1'].min(), D['x1'].max()]
    x2_lims = [D['x2'].min(), D['x2'].max()]

    #Plotting the decision boundaries
    plotBoundaries(decision_boundaries, title, x1_lims, x2_lims)

    ### MSE = MAE for this boolean problem. Technically I'm using MAE.
    er = 0
    for i, row in test.iterrows():

        p = Predict(root, row)
        print(f"Predicted: {p}")
        print(f"Actual: {row['y']}")
        if p != row['y']:
            er+=1

    err = er / len(test) 
    print(f"Error for sub = {sub} with {N} nodes is {err}")
    output.append([sub, N, err])
    
### I just mainly copied the output into a latex table.
print(output)

