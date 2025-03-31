import copy
import json
import re
from pathlib import Path
from utils import tk_popover
from typing import Literal
import numpy as np
from h5py import File  # This is a custom HDF5 format, as far as I know, so I use h5py instead of hyperspy
from matplotlib import pyplot as plt
from scanning_drift_corr.api import SPmerge01linear, SPmerge02, SPmerge03
from tifffile import tifffile
import SingleOrigin as so


def _ftype_check(suffix: str, running: str) -> bool:
    # Compare suffix of current file with the running suffix to ensure all files are of the same type
    if suffix == "":
        return False  # This means that the "file" whose suffix we're checking is either a directory or has no extension
    if running == "":
        return True  # Always okay to overwrite empty (default) extension
    if suffix == running:
        return True  # This is the expected behavior for a set of files that all have the same suffix
    else:
        return False  # suffix != running, or else something weird went wrong and we want to error out anyway


def _minmax_norm(img: np.ndarray) -> np.ndarray:
    _min, _max = np.min(img), np.max(img)
    normed = (img - _min) / (_max - _min)
    return normed


# %% Load data
get_dpc_images: bool = True  # Whether to search for DPC scans in the .h5 file; does nothing with loose files
scan_angles: np.ndarray = np.array([0, 90, 180, 270])  # List of scan rotations per (loose) file; does nothing with .h5
num_frames_to_use: int = 2  # The number of frames to use in drift correction (usually either 2 or 4)
cal: str = 'NOCAL'  # Default export filename component if no scan calibration used

structure_type: Literal["h5", "loose"] = "h5"  # Other format types may be supported in the future
match structure_type:
    case "h5":
        fpath = Path(tk_popover(filetypes=[("HDF 5", ".h5")]))
        fname = fpath.stem
        with File(fpath.resolve(strict=True), 'r') as f:
            # Extract scan angles, then take the first few set by num_frames_to_use (or warn that there are fewer)
            # noinspection PyRedeclaration
            scan_angles = np.array(f['Frame Rotation'])  # This overwrite is intentional
            if num_frames_to_use <= scan_angles.shape[0]:
                scan_angles = scan_angles[:num_frames_to_use]
            else:
                raise ValueError(f"Fewer scan frames available ({scan_angles.shape[0]}) than"
                                 f" requested ({num_frames_to_use}): reduce `num_frames_to_use`")

            # Extract the HAADF scans and, if desired and available, extract the DPC scans as well
            haadfs = np.array(f['Images']['HAADF'][re.search(r"'(.*)'",
                                                             repr(str(f['Images']['HAADF'].keys()))).group(1)])
            if get_dpc_images:
                if np.isin('Segment1 - Segment3', list(f['Images'].keys())):
                    dpc_found = True
                    # Load DPC difference signals
                    dpc_ac = np.array(f['Images']['Segment1 - Segment3']
                                      [re.search(r"'(.*)'", repr(str(f['Images']['Segment1 - Segment3'].keys())))
                                       .group(1)])
                    dpc_bd = np.array(f['Images']['Segment2 - Segment4']
                                      [re.search(r"'(.*)'", repr(str(f['Images']['Segment2 - Segment4'].keys())))
                                      .group(1)])
                else:
                    dpc_found = False
            else:
                dpc_found = False

                # Finally, get the metadata
            metadata = json.loads(f['Metadata'].asstr()[0])
    case "loose":
        file_list = tk_popover(many=True, filetypes=[("TIFF", ".tif"), ("PNG", ".png"), ("JPEG", ".jpg"),
                                                     ("Bitmap", ".bmp"), ("GIF", ".gif")])
        file_list = [Path(file) for file in file_list]
        # All files must have the same suffix, and be of a known type
        supported = {".tif", ".png", ".jpg", ".bmp", ".gif"}
        ftype_check = ""
        for f in file_list:
            if _ftype_check(f.suffix, ftype_check):
                if f.suffix in supported:
                    ftype_check = f.suffix
            else:
                raise RuntimeError("All loose files must be of the same type")

        print([f"{i}: {f.stem}" for i, f in enumerate(file_list)])  # Verify that things are in the right order
        # If the files are in the wrong order, you can either change the order of file_list or of scan_angles

        fname = file_list[0].stem  # Used later on for saving
        ftype = file_list[0].suffix  # We just guaranteed that all files are always the same type
        match ftype:
            case ".tif":
                load_fn = tifffile.imread
            case _:
                raise NotImplementedError(f"File formats other than .tif are not yet supported.  "
                                          f"Supported filetypes will eventually include: {supported}")

        haadfs = []
        for file in file_list:
            with open(file, "rb") as f:
                # noinspection PyTypeChecker
                haadfs.append(load_fn(f))
        haadfs = np.array(haadfs)
        haadfs = np.moveaxis(haadfs, 0, 2)

        # Loose files won't contain extra DPC data, nor metadata
        dpc_found = False
        metadata = {"Pixel Size [nm]": (0, 0),
                    "Convergence Semi-angle [mrad]": 0}
    case _:
        raise RuntimeError(f"Unrecognized load structure: {structure_type}")


# %% Rotate and normalize images for scanning_drift_corr
haadfs = np.array([np.rot90(haadfs[:, :, i], -(a//90)) for i, a in enumerate(scan_angles)])
haadfs = so.image_norm(haadfs[:num_frames_to_use])
if dpc_found:
    dpc_ac = np.array([so.fast_rotate_90deg(dpc_ac[:, :, i], -ang)
                       for i, ang in enumerate(scan_angles)])
    dpc_bd = np.array([so.fast_rotate_90deg(dpc_bd[:, :, i], -ang)
                       for i, ang in enumerate(scan_angles)])
    dpc_ac = so.image_norm(dpc_ac)
    dpc_bd = so.image_norm(dpc_bd)
    dpc_mag = (dpc_ac**2 + dpc_bd**2)**0.5

# Extract microscope parameters from the metadata
pixel_size = float(metadata['Pixel Size [nm]'][0])/1e-9
conv_angle = float(metadata['Convergence Semi-angle [mrad]'])*1e-3

# %% Display the first two images to make sure everything looks right
fig, axs = plt.subplots(1, 2)
axs[0].imshow(haadfs[0], cmap="binary")
axs[1].imshow(haadfs[1], cmap="binary")
plt.show()

# %% Initial correction - linear only
sm = SPmerge01linear(scan_angles, haadfs, niter=1)

# %% Non-linear correction
sm2 = SPmerge02(sm, 16, 8, flagGlobalShift=True, stepSizeReduce=0.5)

# %% Final merge, and apply correction to DPC components (if dpc_found)
haadf_drift_corr, _, _ = SPmerge03(sm2, KDEsigma=0.5)
if dpc_found:
    sigma = 0.5

    sm_ac = copy.deepcopy(sm2)
    # noinspection PyUnboundLocalVariable
    sm_ac.scanLines = dpc_ac
    image_final_ac, _, _ = SPmerge03(sm_ac, KDEsigma=sigma)

    sm_bd = copy.deepcopy(sm2)
    # noinspection PyUnboundLocalVariable
    sm_bd.scanLines = dpc_bd
    image_final_bd, _, _ = SPmerge03(sm_bd, KDEsigma=sigma)

# %% Save and export to tiff
directory = Path(tk_popover(open_dir=True))

tifffile.imwrite(directory / f"{fname}_HAADF_DCORR_{cal}.tif", haadf_drift_corr, photometric="minisblack")
if dpc_found:
    # noinspection PyUnboundLocalVariable
    tifffile.imwrite(directory / f"{fname}_DPC-AC_DCORR_{cal}.tif", image_final_ac, photometric="minisblack")
    # noinspection PyUnboundLocalVariable
    tifffile.imwrite(directory / f"{fname}_DPC-BD_DCORR_{cal}.tif", image_final_bd, photometric="minisblack")
