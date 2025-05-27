from scipy import stats
import pandas as pd
from utils import tk_popover

# %%
indata = pd.read_excel(tk_popover())

# %%
d = indata["Al/AlN"][1:].values

al_exp = indata["Unnamed: 1"][1:].values
al_rand = indata["Unnamed: 2"][1:].values

sc_exp = indata["Unnamed: 5"][1:].values
sc_rand = indata["Unnamed: 6"][1:].values

# %%
ks2_al = stats.ks_2samp(al_exp, al_rand)
ks2_sc = stats.ks_2samp(sc_exp, sc_rand)

# %%
print(ks2_al)
print(ks2_sc)

#%%
