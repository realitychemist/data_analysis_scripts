from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import scipy
import pandas
import matplotlib.pyplot as plt

def tk_popover(save: bool = False, **kwargs):
    """Tk helper to ensure window appears on top."""
    root = Tk()
    root.iconify()
    root.attributes('-topmost', True)
    root.update()
    loc = None  # Default return if open fails; will likely cause an error when passed along
    try:
        if not save:
            loc = askopenfilename(parent=root, **kwargs)
        else:
            loc = asksaveasfilename(parent=root, **kwargs)

    finally:
        root.attributes('-topmost', False)
        root.destroy()
    return loc

# %% Read in EELS data from .xlsx
indata = pandas.read_excel(tk_popover())

# %% Select the correct columns
E, I = indata.Energy, indata.total

# %% Broaden & plot
"""Sigma is defined by (FWHM/(2sqrt(2ln2))/dispersion. For a FWHM of 1.45 eV and dispersion of 0.1 eV/channel,
this corresponds to 0.616/0.1"""
broad = scipy.ndimage.gaussian_filter1d(I, sigma=0.616/0.1, mode="nearest")

plt.plot(E.values, I.values, color="orange", lw=1, linestyle="--", label="Original")
plt.plot(E.values, broad, color="green", lw=1, linestyle="-", label="Broadened")
plt.xlim((401, 436))
plt.gca().set_yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity")
plt.legend()
plt.show()

#%% Save
df_broad = pandas.DataFrame({"E": E, "I": broad})
with pandas.ExcelWriter(tk_popover(), mode="a") as writer:
    df_broad.to_excel(writer, index=False, sheet_name="broadened")
#%%
