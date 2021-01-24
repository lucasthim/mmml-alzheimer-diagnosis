# %%
import pandas as pd

# %%
df_mri_meta = pd.read_csv("MRIMETA.csv")
df_mri_meta.shape

# %%
df_mri_meta

# %%
print('columns: ',df_mri_meta.columns)

# %%
df_mri_meta.sort_values("RID").iloc[:10,:15]

# %%
df_mri_meta["PHASE"].unique()

#%%
df_tadpole_d1d2 = pd.read_csv('tadpole_challenge/TADPOLE_D1_D2.csv')
df_tadpole_d1d2.shape

# %%
df_tadpole_d1d2.columns[:100]

# %%
cols_d1d2 = df_tadpole_d1d2.columns.to_list()
filtered_cols = [x for x in cols_d1d2 if "UCSFFSL" in x]
filtered_cols

# %%
df_tadpole_d1d2[filtered_cols].iloc[:10]

# %%
df_tadpole_d1d2[["RID"] + [x for x in df_tadpole_d1d2.columns if ("PET" in x) or ("UCBERKELE" in x)]]

