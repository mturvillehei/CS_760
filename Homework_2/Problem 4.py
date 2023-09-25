#~~~~ Morgan Turville-Heitz ~~~~#
#~~~~ 09/24/2023 ~~~~#
#~~~~ CS 760 Fall 2023 ~~~~#

import pandas as pd
import math
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

# Over the interval a,b sample 100 points in x, and let y = sin(x).

a = 0.001
b = math.pi
n = 100

### Generating my noise
e = np.linspace(0.0, 20, 40)

### Iterating over noise
for eps in e:
    print(f"stddev is {eps}")

    ## Generating my training set
    x_ = np.linspace(a, b, n)
    
    ### In this case, adding noise after y = sin(x).
    y_ = np.sin(x_)
    for i, x in enumerate(x_):
        x_[i] = x + np.random.normal(loc=0, scale=eps)
    ## Generating my test set
    xt = np.linspace(a,b, 100)
    for i, x in enumerate(xt):
        xt[i] = x + np.random.normal(loc=0,scale=eps)
    yt = np.sin(xt)

    #Building the model f
    poly =  lagrange(x_, y_)

    ###Using MSE
    err = np.mean((y_ - poly(x_))**2)
    print(f"Training error is: {err}")

    ###Using MSE
    err_t = np.mean((yt - poly(xt))**2)
    print(f"Test error is: {err_t}")
