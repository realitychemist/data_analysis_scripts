from tkinter import Tk
import numpy as np
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


def lorentzian(x, x0, gamma):
    return gamma / (np.pi * ((x-x0)**2 + gamma**2))


# %% Read in EELS data from .xlsx
indata = pandas.read_excel(tk_popover(), sheet_name="interp")

# %% Select the correct columns
E, I = indata["E_Interp"], indata["I_Avg"]

# %% Broaden & plot
fwhm: float = 1.3  # User-specified FWHM (eV, as measured from ZLP)

# We must ensure equi-spacing in E for convolution to work correctly
E_grid = np.linspace(min(E.values), max(E.values), len(E))
interp_function = scipy.interpolate.interp1d(E, I, kind="linear")
I_interp = interp_function(E_grid)

if len(E_grid)%2 == 0:
    center = E_grid[len(E_grid)//2-1]
else:
    center = E_grid[len(E_grid)//2]

lor = lorentzian(E_grid, center, fwhm/2)
lor /= np.sum(lor)  # Normalize

broad = scipy.signal.convolve(I_interp, lor, mode="same")

# %% Plot preview
plt.plot(E.values, I.values, color="orange", lw=1, linestyle="--", label="Original")
plt.plot(E_grid, broad, color="green", lw=1, linestyle="-", label="Broadened")
plt.xlim((401, 436))
plt.gca().set_yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity")

plt.legend()
plt.show()

# %% Save
df_broad = pandas.DataFrame({"E": E_grid, "I": broad})
with pandas.ExcelWriter(tk_popover(), mode="a") as writer:
    df_broad.to_excel(writer, index=False, sheet_name="broadened")
print("Saved!")
