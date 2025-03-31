"""Dr Probe simulated 4D datasets can be exported as tiff stacks, but py4DSTEM cannot import a tiff stack as a proper
4D-STEM datacube. This short script handles loading of the .tif file, its conversion to the proper shape (using
metadata stored in the Dr Probe metadata .json file), and its export as a py4DSTEM-native .emd file."""

import json
import emdfile
import tifffile
import py4DSTEM
import numpy as np
import os.path as path
from tempfile import TemporaryDirectory
from timeit import default_timer
from utils import tk_popover

crop_and_bin = True  # Set to true to crop-and-bin data before saving, or False to save the original data
crop_to: tuple[int, int] = (600, 600)  # Size to crop individual diffraction patterns to (before binning)
bin_factor: int = 3  # Number of pixels to bin in each image direction

# %% Open JSON meta-file
meta_fname = tk_popover(filetypes=[("JSON Files", ".json")])
with open(meta_fname, "r") as f:
    input_meta = json.load(f)

# %% Open tiff stack
tiff_fname = tk_popover(filetypes=[("Tagged Image File Format", ".tif")])
input_tiff = tifffile.memmap(tiff_fname)  # Memory map needed for very large files

# %% Decide where the datacube will be saved
emd_fname = tk_popover(save=True, defaultextension=".emd", filetypes=[("Electron Microscopy Dataset", ".emd")])

# %% Reshape, optionally crop-and-bin, then save
# Reshaping the memory map is fast
scan_x = input_meta["scan"]["dims"][0]
scan_y = input_meta["scan"]["dims"][1]
diff_kx = input_meta["diffraction"]["dims"][0]
diff_ky = input_meta["diffraction"]["dims"][1]
shaped_data = input_tiff.reshape((scan_x, scan_y, diff_kx, diff_ky))

# Crop and bin down the data before saving
with TemporaryDirectory() as workdir:
    print(f"Working in temporary directory at {workdir}; if the program crashes, make sure to go delete it manually!")
    if crop_and_bin:
        print("Cropping & binning...")
        time_precb = default_timer()

        slice_kx = slice(diff_kx//2 - crop_to[0]//2, diff_kx//2 + crop_to[0]//2)
        slice_ky = slice(diff_ky//2 - crop_to[1]//2, diff_ky//2 + crop_to[1]//2)
        cropshape = (shaped_data.shape[0], shaped_data.shape[1], crop_to[0], crop_to[1])
        cropped_data = np.memmap(path.join(workdir, "_crp.dat"),
                                 mode="w+", dtype=input_meta["data_type"], shape=cropshape)
        cropped_data[:] = shaped_data[:, :, slice_kx, slice_ky]  # Crop in toward center symmetrically

        # We bin down the data by reshaping into 6D, then taking the mean along the dummy dimensions
        binshape_kx, binshape_ky = cropped_data.shape[2] // bin_factor, cropped_data.shape[3] // bin_factor
        binning_shape = (cropped_data.shape[0], cropped_data.shape[1],
                         binshape_kx, cropped_data.shape[2]//binshape_kx,
                         binshape_ky, cropped_data.shape[3]//binshape_ky)
        binned_shape = (cropped_data.shape[0], cropped_data.shape[1], binshape_kx, binshape_ky)
        binned_data = np.memmap(path.join(workdir, "_bin.dat"),
                                mode="w+", dtype=input_meta["data_type"], shape=binning_shape)
        binned_data[:] = cropped_data.reshape(binning_shape)
        dataset = np.memmap(path.join(workdir, "_avg.dat"),
                            mode="w+", dtype=input_meta["data_type"], shape=binned_shape)
        dataset[:] = binned_data.mean(5).mean(3)
        time_postcb = default_timer()
        print(f"Done! Cropping & binning took {time_postcb-time_precb:.0f} seconds.")
    else:
        dataset = shaped_data

    print("Saving...")
    time_presave = default_timer()

    emd_data = emdfile.Array(dataset, name="intensity")

    calibration = py4DSTEM.Calibration()
    calibration.set_R_pixel_units(input_meta["scan"]["units"])
    calibration.set_R_pixel_size(input_meta["scan"]["step"][0] * bin_factor)
    calibration.set_Q_pixel_units("A^-1")  # Annoyingly py4DSTEM won't allow nm^-1 units
    calibration.set_Q_pixel_size(input_meta["diffraction"]["step"][0] * bin_factor / 10)  # Convert nm^-1 to A^-1

    datacube = py4DSTEM.DataCube(emd_data.data, calibration=calibration)
    py4DSTEM.save(emd_fname, emd_data)

    time_postsave = default_timer()
    print(f"Done! Saving took {time_postsave-time_presave:.0f} seconds.")

#%%
