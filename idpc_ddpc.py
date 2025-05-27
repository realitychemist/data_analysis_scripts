"""Process DPC data from an .emd (Velox) file into iDPC/dDPC images.
Script by CSE, adapted from a script by scaldero."""
from pathlib import Path
from data_analysis_scripts.utils import tk_popover
from tkinter.simpledialog import askfloat
from tifffile import imwrite
import hyperspy.api as hs
import numpy as np


# %% Load the images from .emd
infile = hs.load(tk_popover())
try:
    img_ac = next(signal for signal in infile if signal.metadata.General.title == "A-C")
    img_bd = next(signal for signal in infile if signal.metadata.General.title == "B-D")
except StopIteration:
    try:  # Might have individual sectors; construct difference images
        img_a = next(signal for signal in infile
                     if signal.metadata.General.title == "DF4-A")
        img_b = next(signal for signal in infile
                     if signal.metadata.General.title == "DF4-B")
        img_c = next(signal for signal in infile
                     if signal.metadata.General.title == "DF4-C")
        img_d = next(signal for signal in infile
                     if signal.metadata.General.title == "DF4-D")
        img_ac = img_a.data-img_c.data
        img_bd = img_b.data-img_d.data
    except StopIteration:
        raise RuntimeError("File may not contain any DPC data")
# haadf = _minmax_norm(next(signal for signal in infile
#                           if signal.metadata.General.title == "HAADF"))

# %% Calculate the iDPC & dDPC images
lowstop = 0.1  # Filter out low frequencies; must be > 0

try:
    rot_angle = float(img_ac.original_metadata.CustomProperties.DetectorRotation.value) + np.pi
except AttributeError:  # Original file might have contained individual sector images
    try:
        # noinspection PyUnboundLocalVariable
        rot_angle = float(img_a.original_metadata.CustomProperties.DetectorRotation.value) + np.pi
    except NameError:
        rot_angle = askfloat("Detector Rotation Angle",
                             "No detector rotation record, please input manually (radians):",
                             initialvalue=0, minvalue=0, maxvalue=2*np.pi)


dpc_x = img_ac * np.cos(rot_angle) - img_bd * np.sin(rot_angle)
dpc_y = img_ac * np.sin(rot_angle) + img_bd * np.cos(rot_angle)

# Mesh for k-space based on the size of the DPC_x data; DPC_y must be the same shape
fourier_mesh = np.meshgrid(np.fft.fftfreq(dpc_x.data.shape[-1], 1),
                           np.fft.fftfreq(dpc_x.data.shape[-2], 1))
# Magnitude of k at every point in the meshgrid
k_mag = np.hypot(*fourier_mesh)
# Angle of k at every point in the meshgrid; x and y reversed by convention
k_dir = np.arctan2(*reversed(fourier_mesh))
# Lowstop filtering - 0 frequency filtering is *required*, but more filtering may be desirable
k = np.where(k_mag > lowstop, k_mag, 0)
# k_sq == k**2, except inside the lowstop where it is clamped to 1
k_sq = np.where(k > 0, k ** 2, 1)
# The actual iDPC math, for both x and y; see doi: 10.1038/s41598-018-20377-2
idpc_x = np.absolute(np.fft.ifft2((np.fft.fft2(dpc_x) * k_mag * np.cos(k_dir)) /
                                  (2 * np.pi * 1j * k_sq)))
idpc_y = np.absolute(np.fft.ifft2((np.fft.fft2(dpc_y) * k_mag * np.sin(k_dir)) /
                                  (2 * np.pi * 1j * k_sq)))
idpc = (idpc_x + idpc_y).astype("float32")

# Calculate the dDPC image
grad_x = np.gradient(dpc_x, axis=-1)
grad_y = np.gradient(dpc_y, axis=-2)
ddpc = -(grad_x + grad_y).astype("float32")

# %% Export the images
export_registered: bool = True
directory = Path(tk_popover(open_dir=True))
# By default this script saves the new images with the same filename as the original file
fname = infile[0].tmp_parameters.original_filename
# Optionally, uncomment to provide a new filename as a string (no extension)
# fname = "test_filename"

imwrite(directory / (fname + "_iDPC.tif"), idpc, photometric="minisblack")
imwrite(directory / (fname + "_dDPC.tif"), ddpc, photometric="minisblack")

#%%
