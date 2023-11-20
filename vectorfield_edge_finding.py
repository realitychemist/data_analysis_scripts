from pathlib import Path
from tkinter.filedialog import askopenfilename, askdirectory
import hyperspy.api as hs
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imwrite


def _vector_gradient(vecfield: np.ndarray, idx: tuple[int, int])\
        -> tuple[tuple[float, float], tuple[float, float]] | tuple[tuple[None, None], tuple[None, None]]:
    """Compute the vector gradient at a point in a 2D array.
    Args:
        vecfield: The entire vector field.
        idx: The 2D index of the point at which we want to compute the vector gradient.

    Returns: The local vector gradient tensor, in the form described at:
    https://math.stackexchange.com/questions/156880/what-does-it-mean-to-take-the-gradient-of-a-vector-field
    For entries on the edge of the array, returns None.
    """
    x, y = idx[0], idx[1]
    xlim, ylim, _ = vecfield.shape

    # Edge check
    if x == 0 or y == 0 or x == xlim-1 or y == ylim-1:
        return (None, None), (None, None)

    lfvec = vecfield[x-1, y]
    rtvec = vecfield[x+1, y]
    upvec = vecfield[x, y-1]
    dnvec = vecfield[x, y+1]

    dx1 = float((lfvec[0] - rtvec[0]) / 2)
    dx2 = float((lfvec[1] - rtvec[0]) / 2)
    dy1 = float((upvec[0] - dnvec[0]) / 2)
    dy2 = float((upvec[1] - dnvec[1]) / 2)

    return (dx1, dx2), (dy1, dy2)


# TODO: We might only want to use the g00 component (df1/dx) to isolate the vertical edges instead of summing in g11
def _trace(grad_tensor: np.array)\
        -> float | None:
    g00 = grad_tensor[0, 0]
    g11 = grad_tensor[1, 1]
    if g00 is None or g11 is None:
        return None
    else:
        return g00 + g11


def _minmax_norm(img):
    _min, _max = np.min(img), np.max(img)
    normed = (img - _min) / (_max - _min)
    return normed


# %%
infile = hs.load(askopenfilename())

# %%
dpc_img = (next(signal for signal in infile if signal.metadata.General.title == "DPC"))
complex_data = dpc_img.data
vector_data = np.array([[(np.real(entry), np.imag(entry)) for entry in row] for row in complex_data])

# %%
xs, ys = np.arange(0, vector_data.shape[0]), np.arange(0, vector_data.shape[1])
tensor_data = np.array([_vector_gradient(vector_data, (i, j))
                        for i in xs for j in ys]).reshape(vector_data.shape[0],
                                                          vector_data.shape[1], 2, 2)
# %%
divergences = np.array([_trace(tensor) for row in tensor_data for tensor in row]).reshape(tensor_data.shape[0],
                                                                                          tensor_data.shape[1])
divergences = _minmax_norm(np.array([0 if div is None else div
                                     for row in divergences for div in row])).reshape(tensor_data.shape[0],
                                                                                      tensor_data.shape[1]).astype("float32")

# %%
plt.imshow(divergences, cmap="binary", vmin=0.3, vmax=0.6)

# %%
directory = Path(askdirectory())
fname = infile[0].tmp_parameters.original_filename
imwrite(directory / (fname + "_divergence_map.tif"), divergences, photometric="minisblack")

#%%
