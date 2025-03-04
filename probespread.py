import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, asksaveasfilename


def tk_popover(save: bool = False, **kwargs):
    """Tk helper to ensure window appears on top."""
    root = Tk()
    root.iconify()
    root.attributes('-topmost', True)
    root.update()
    loc = None  # Default return if open fails; will likely cause an error when passed along
    try:
        if not save:
            loc = askdirectory(parent=root, **kwargs)
        else:
            loc = asksaveasfilename(parent=root, **kwargs)

    finally:
        root.attributes('-topmost', False)
        root.destroy()
    return loc


# %%
src = Path(tk_popover())
filelist = sorted(src.glob("*.tif"))

# %%
depthimg = []
for f in filelist:
    tif = tifffile.imread(f)
    depthimg.append(tif.sum(axis=0))
depthimg = np.array(depthimg, dtype="float32")
plt.imshow(depthimg, cmap="inferno")

#%%
mode = "linear"
x_scale: float = 3.66564  # nm
z_scale: float = 5.31  # nm
power_exp = 0.5  # Used if mode is powerlaw

px_size = x_scale/len(depthimg[0])  # nm/px
z_px = round(z_scale/px_size)  # px
scaled_image = cv2.resize(depthimg, dsize=(len(depthimg[0]), z_px), interpolation=cv2.INTER_LINEAR)

match mode:
    case "linear":
        pass
    case "log":
        # Make all values >= 1 to avoid negative values after scaling
        scaled_image = cv2.normalize(scaled_image, None, 1, 2, cv2.NORM_MINMAX)
        scaled_image = np.log(scaled_image, where=(scaled_image!=0))
    case "powerlaw":
        scaled_image = np.power(scaled_image, power_exp)
    case _:
        raise NotImplementedError(f"Mode {mode} not implemented")

plt.imshow(scaled_image, cmap="inferno")
plt.xticks([])
plt.yticks([])
plt.gca().spines[:].set_visible(False)

plt.savefig(tk_popover(save=True), dpi=300, format="png")

#%%
