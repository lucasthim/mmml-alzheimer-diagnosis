# %%
import pandas as pd
import numpy as np
# %%
df_mci = pd.read_csv('./../data/PREDICTIONS_MCI_VGG19_BN_1125.csv')
# %%
df = df_mci.query("IMAGE_DATA_ID == 'I368981'")
x = df.query("ORIENTATION == 'axial'").iloc[0]
# %%
x.to_dict()
# %%
