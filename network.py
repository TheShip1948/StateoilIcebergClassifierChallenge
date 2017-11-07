####################################################
# --- Import ---
####################################################
import numpy as np 
import pandas as pd 
from os.path import join as opj 
from matplotlib import pyplot as plt 
plt.rcParams['figure.figsize'] = 10, 10

df = pd.read_json(opj('.', 'Data', 'train.json'))
