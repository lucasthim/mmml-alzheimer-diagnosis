# %%
import pandas as pd
import numpy as np
# %%

def main():

    dicts = [
        {'key1':123,'key2':432},
        {'key1':123,'key2':1111},
        {'key1':123,'key2':'dwdw'},
        {'key1':123,'key2':444}
    ]
    pop_key(dicts)
    return dicts

def pop_key(dicts):
    for d in dicts:
        del d['key2']

print(main())
# %%
