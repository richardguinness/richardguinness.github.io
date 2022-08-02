---
layout: post
title:  "A big notebook"
date:   2022-08-02 11:56:18 +0100
categories: jekyll update
---
```python
import json, re, sys

import matplotlib.pyplot as plt, seaborn as sb
import math
import numpy as np, pandas as pd
#import requests
import graphviz # conda install python-graphviz

import scipy
from scipy.cluster import hierarchy as hc

from IPython.display import clear_output

from sklearn import tree, metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import warnings

sb.set(rc={"figure.dpi": 78, "savefig.dpi": 78, "figure.figsize":(10,6)})
sb.set(rc={})
sb.set_context("notebook")
sb.set_style("ticks")
sb.set_style("darkgrid")  # Make it pretty!
base_color = sb.color_palette()[0]

# Ensure we see all columns, and the contents of each!
#pd.set_option("display.max_colwidth", None)
#pd.set_option("display.max_columns", None)

np.set_printoptions(linewidth=130)

%matplotlib inline
# %config InlineBackend.figure_format = 'retina' # Increase resolution of plots. Seems to conflict with seaborn settings above

# automatically reloads imports
%load_ext autoreload
%autoreload 2

DATA_PATH = 'data/'

print(f"sys.version: {sys.version}")
print(f"pd.__version__: {pd.__version__}")
```

    sys.version: 3.10.4 (main, Mar 31 2022, 08:41:55) [GCC 7.5.0]
    pd.__version__: 1.4.3



```python
!pwd
```

    /home/rdzg/Seafile/Learning/kaggle/titanic_1



```python
!ls {DATA_PATH}
```

    gender_submission.csv  test.csv  train.csv



```python
!head -n 1 {DATA_PATH}train.csv
```

    PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked



```python
!shuf -n 3 {DATA_PATH}train.csv
```

    458,1,1,"Kenyon, Mrs. Frederick R (Marion)",female,,1,0,17464,51.8625,D21,S
    376,1,1,"Meyer, Mrs. Edgar Joseph (Leila Saks)",female,,1,0,PC 17604,82.1708,,C
    462,0,3,"Morley, Mr. William",male,34,0,0,364506,8.05,,S


Steps to take:

1. EDA
1. Split the given training data set into Training and Validation sets. Decide on a proportion: 20% of rows for Validation?
1. Drop the 'Survived' col from the Validation set
1. Copy the 'Survived' col from the new Training set to a new 1d array 'y'
1. Dropping the 'Survived' col, copy the Training set to a new 2d array 'X'



[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
