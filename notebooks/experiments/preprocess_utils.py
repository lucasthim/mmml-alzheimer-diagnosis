import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sp
from joblib import dump, load
from collections import namedtuple

%matplotlib inline
plt.style.use('seaborn')
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)


output_path = "/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/preprocessed/20210523/"
input_path='/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/raw/ADNI/'