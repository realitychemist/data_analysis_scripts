# Interpolate spectra which have two different x-axes (energy for EELS, 2-theta for XRD, etc...)
import numpy as np
import pandas as pd
from utils import tk_popover


# %% Read in EELS data from .xlsx
indata = pd.read_excel(tk_popover())

# %%
e_columns = [col for col in indata.columns if col.startswith('E_')]
i_columns = [e_col.replace("E", "I") for e_col in e_columns]
unique_e = np.unique(np.concatenate([indata[col] for col in e_columns]))

# %%
new_df = pd.DataFrame({"E": unique_e})
for e_col, i_col in zip(e_columns, i_columns):
    temp_df = indata[[e_col, i_col]].copy()
    temp_df.rename(columns={e_col: "E"}, inplace=True)
    new_df = pd.merge(new_df, temp_df, on="E", how="outer")
    max_e = max(temp_df["E"])
    new_df[i_col] = new_df[i_col].interpolate(method="linear")
    new_df.loc[new_df["E"] > max_e, i_col] = pd.NA  # Tidy the end of the list

# %%
with pd.ExcelWriter(tk_popover(save=True), mode="a") as writer:
    new_df.to_excel(writer, index=False, sheet_name="interp")
