from pathlib import Path
from tkinter.filedialog import askopenfilename, asksaveasfilename
from itertools import combinations
from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import quickcrop as qc
import SingleOrigin as so
import matplotlib
from tifffile import imread
from importlib.metadata import version
import hyperspy.api as hs
from scipy.stats import describe
from warnings import warn
# from mp_api.client import MPRester
# from pymatgen.io.ase import AseAtomsAdaptor
# from data_analysis_scripts.cse_secrets import MP_API_KEY

if version("hyperspy") > "1.7.6":
    warn("File IO was deprecated in HyperSpy versions 2.0rc0 and beyond; .emd file importing will not work.")

matplotlib.use("TkAgg")
plt.style.use('dark_background')


def _minmax_norm(img):
    _min, _max = np.min(img), np.max(img)
    normed = (img - _min) / (_max - _min)
    return normed


def nn_distances(origin_lat: np.ndarray, dest_lat: np.ndarray, pixel_size: float)\
        -> np.ndarray:
    # position array column order is: x_fit, y_fit, u, v
    distances = []
    for origin in origin_lat:
        u, v = origin[2], origin[3]
        # Get nearest neighbors
        # TODO: For non-orthogonal lattices, this rook-style neighborhood might not always return all nearest neighbors
        # (a proper solution probably involves using a kD tree to find all neighbors with a given cutoff,
        #  but that approach will be both more complex and, probably, slower)
        if origin_lat is dest_lat:
            a = dest_lat[(dest_lat[:, 2] == u+1) & (dest_lat[:, 3] == v), :2]
            b = dest_lat[(dest_lat[:, 2] == u-1) & (dest_lat[:, 3] == v), :2]
            c = dest_lat[(dest_lat[:, 2] == u) & (dest_lat[:, 3] == v+1), :2]
            d = dest_lat[(dest_lat[:, 2] == u) & (dest_lat[:, 3] == v-1), :2]
        else:  # TODO: This assumes a 100 perovskite; any good way to generalize to other axes / structures?
            a = dest_lat[(dest_lat[:, 2] == u+0.5) & (dest_lat[:, 3] == v+0.5), :2]
            b = dest_lat[(dest_lat[:, 2] == u+0.5) & (dest_lat[:, 3] == v-0.5), :2]
            c = dest_lat[(dest_lat[:, 2] == u-0.5) & (dest_lat[:, 3] == v-0.5), :2]
            d = dest_lat[(dest_lat[:, 2] == u-0.5) & (dest_lat[:, 3] == v+0.5), :2]

        distances.extend([np.linalg.norm(origin[:2] - neighbor_pos)*pixel_size
                          for neighbor_pos in [a, b, c, d]
                          if not len(neighbor_pos) == 0])
    return np.asarray(distances)


# %% Import the image
crop: bool = True  # Set this to true to interactively crop image after loading
path = Path(askopenfilename())
match path.suffix:
    case ".tif":
        # noinspection PyTypeChecker
        img = _minmax_norm(np.array(imread(path)))
    case ".emd":
        infile = hs.load(path)
        # The only kind of data it makes sense to load from an .emd is from the HAADF detector
        # iDPC / dDPC images are not computed and saved in the .emd file
        img = _minmax_norm(next(signal for signal in infile
                                if signal.metadata.General.title == "HAADF"))
    case "":
        raise RuntimeError("No type extension in file name; either file has no extension or no file was selected.")
    case _:
        raise RuntimeError(f"Unsupported filetype: {path.suffix}")

if crop:
    img = qc.gui_crop(img)

# %% Import the structure file
load_method: Literal[
    "fixed",  # Load via a fixed file path, given below, without a file picker window
    "interactive",  # Interactively select the file using a (native) file picker window
    "mp_api"  # Load a cif from Materials Project; requires an API key!
    ] = "fixed"

fixed_path = Path("E:/Users/Charles/BZT.cif")  # Local path to .cif file
mp_id = "mp-661"  # Materials Project ID for desired structure, including the leading "mp-"

origin_shift: tuple[float, float, float] = (0, 0, 0)  # Set to shift the unit cell origin of the structure on load
# To relabel atomic sites in the unit cell on load, set as a dict from default label to new label
# If there are no sites to relabel, leave the dictionary empty
# This is mainly cosmetic (affect label in vPCF plots)
replacement_labels: dict[str, str] = {"Ti": "Ti/Zr"}

match load_method:
    case "fixed":
        uc = so.UnitCell(str(fixed_path), origin_shift=origin_shift)
    case "interactive":
        interactive_path = Path(askopenfilename())
        uc = so.UnitCell(str(interactive_path), origin_shift=origin_shift)
    case "mp_api":
        raise NotImplementedError("Loading from the MP API is not yet implemented in this script!")
        # TODO: UnitCell wants a string pointing to a local file; might need to copy the cif to a temp file
        # mp_api_key = MP_API_KEY  # Personal API key (tied to MP account)
        #
        # with MPRester(mp_api_key) as mp:
        #     mp_struct = mp.get_structure_by_material_id(mp_id)
        #     print(mp_struct)
        #     struct = AseAtomsAdaptor().get_atoms(mp_struct)
    case _:
        raise NotImplementedError(f"Unrecognized file load method: {load_method}")

for key, replacement in replacement_labels.items():
    uc.atoms.replace(key, replacement, inplace=True)

# %% Project the unit cell along a given zone axis
plot_projected_cell: bool = True  # Set True to pop up a visualization of the projected unit cell for verification

uc.project_zone_axis((0, 0, 1),  # Zone axis direction
                     (1, 0, 0),  # Apparent horizontal axis in projection
                     (0, 1, 0),  # Most vertical axis in projection
                     ignore_elements=["O"]  # List of elements to ignore (e.g. light elements in a HAADF image)
                     )
uc.combine_prox_cols(toler=1e-2)  # 1e-2 works well as a tolerance, but adjust as needed

if plot_projected_cell:
    uc.plot_unit_cell()

# %% Create the SingleOrigin HRImage object
hr_img = so.HRImage(img)
# noinspection PyTypeChecker
lattice = hr_img.add_lattice("BZT",  # Human-readable name for the lattice in the image
                             uc,
                             origin_atom_column=None)  # Index in uc of column to use as fitting origin, None == default

# %% Reciprocal-space rough fit of lattice
# Both of these methods spawn windows that must be interacted with before they time out!
# Rerun the cell if you accidentally let them time out
# If some FFT peaks are weak or absent (such as forbidden reflections), specify the order of the first peak that
# is clearly visible
lattice.fft_get_basis_vect(a1_order=1,  # Order of peak corresponding to planes in the most horizontal direction
                           a2_order=1)  # Order of peak corresponding to planes in the most vertical direction
lattice.define_reference_lattice()

# %% Fit the atom columns
lattice.fit_atom_columns(buffer=10,  # Pixel buffer around image edges to avoid partial columns
                         local_thresh_factor=0.0,  # Watershed segmentation threshold
                         use_background_param=True,  # Fit the background intensity
                         use_bounds=True,  # Bound allowed column fit position
                         use_circ_gauss=False,  # Force gaussians to be circular (major axis == minor axis)
                         parallelize=True,  # Parallelize the computation
                         peak_grouping_filter="auto",  # Apply a filter for simultaneous column fitting
                         peak_sharpening_filter="auto")  # Apply a filter to improve column position fitting

# Must have only one column per projected unit cell.  If no sublattice meets this criteria, specify a specific column
# in the projected cell.
lattice.refine_reference_lattice(filter_by="elem",  # Which dataframe column to select on
                                 sites_to_use="Ba")  # Which entry to select

# %% Assess fit quality
# All of these steps are optional but can help with assessing a failed fit; uncomment desired lines
# See function documentation for param descriptions
assess_masks: bool = False
assess_positions: bool = False
assess_residuals: bool = False
assess_displacement: bool = False

if assess_masks:
    lattice.show_masks()
if assess_positions:
    hr_img.plot_atom_column_positions(scatter_kwargs_dict={"s": 20}, scalebar_len_nm=None, fit_or_ref="fit")
if assess_residuals:
    _ = lattice.get_fitting_residuals()
if assess_displacement:
    hr_img.plot_disp_vects(sites_to_plot=["Ba"], arrow_scale_factor=2)

# %% Setup vPCFs
hr_img, _ = hr_img.rotate_image_and_data("BZT", "a1", "horizontal")
lattice = hr_img.latt_dict["BZT"]
pair_pair: bool = True  # If true, will get vPCFs within _and_ between sublattices, otherwise only within
pxsize: float = 0.01  # Angstrom
column_labels: set = {"Ba", "Ti/O"}  # Which columns to generate vPCFs for (usually element names)
xlimits, ylimits = (-1.05, 1.05), (-1.05, 1.05)  # Limits for vPCF plotting (unit cells)

lattice.get_vpcfs(xlim=xlimits, ylim=ylimits, d=pxsize,
                  get_only_partial_vpcfs=(not pair_pair),
                  sublattice_list=list(column_labels))

combos = list(lattice.vpcfs.keys())
combos.remove("metadata")  # Metadata key present in lattice.vpcfs shouldn't be included
combos = [tuple(string.split("-")) for string in combos]

# %% Plot the vPCFs
# Custom plotting code; you can also just use the plotting implemented in SO if you run the peak fitting first
plot_ref: bool = True  # If true, plot reference lattice points
# Adjust min and max values to get the desired level of color saturation on the vPCFs
minval: int = 0
maxval: int = 1000

# These colors come from the IBM colorblind palette and should have okay contrast even in grayscale
basic_colors = {"green":   "#117733ff",
                "magenta": "#dc267fff",
                "orange":  "#fe6100ff"}
# Three colors should be sufficient for two elements with self-vPCFs enabled, or three without; more vPCFs on one
#  plot is probably a bad idea but if you want to do that you'll need more colors
if len(combos) > len(basic_colors):
    raise RuntimeError(f"Trying to plot {len(combos)} vPCFs with {len(basic_colors)} colors: plot fewer vPCFs or add"
                       " additional colors.")
cmaps = [LinearSegmentedColormap.from_list(f"black-{cname}-white",
                                           ["#00000000", cval, "#ffffffff"], 256)
         for cname, cval in basic_colors.items()]

origin_x, origin_y = lattice.vpcfs["metadata"]["origin"]
if plot_ref:  # Generate the reference vPCFs if needed
    ref_coord_dict = {lab: lattice.at_cols.loc[lattice.at_cols["elem"] == lab, "x_ref":"y_ref"].to_numpy()
                      * lattice.pixel_size_est for lab in column_labels}
    ref_xlimits = (xlimits[0]*lattice.a_2d[0][0], xlimits[1]*lattice.a_2d[0][0])
    ref_ylimits = (ylimits[0]*lattice.a_2d[1][1], ylimits[1]*lattice.a_2d[1][1])
    ref_vpcfs = {combo: so.v_pcf(coords1=ref_coord_dict[combo[0]],
                                 coords2=ref_coord_dict[combo[1]],
                                 xlim=ref_xlimits, ylim=ref_ylimits, d=pxsize)
                 for combo in combos}

vpcf_fig, vpcf_ax = plt.subplots()
for vpcf, cmap in zip([value for key, value in lattice.vpcfs.items() if key != "metadata"], cmaps):
    vpcf_ax.imshow(vpcf, cmap=cmap, vmin=minval, vmax=maxval, interpolation=None)
if plot_ref:
    # noinspection PyUnboundLocalVariable
    for ref_vpcf in ref_vpcfs.values():
        vpcf_ax.imshow(ref_vpcf[0], cmap=LinearSegmentedColormap.from_list("transparent-blue",
                                                                           ["#00000000", "#22aaffff"], 2),
                       vmax=1, vmin=0, interpolation="gaussian")

vpcf_ax.plot(origin_x, origin_y, "wx")  # Plot the origin as an "x"
vpcf_ax.set_title("Vector Partial Pair Correlation Functions for "
                  f"{' & '.join(filter(None, [', '.join(list(column_labels)[:-1]), list(column_labels)[-1]]))}")
legend_elements = [Patch(facecolor=c, edgecolor=c, label=f"{combo[0]}–{combo[1]}")
                   for c, combo in zip(basic_colors, combos)]
legend_elements.append(Patch(facecolor="#22aaffff", edgecolor="#22aaffff", label="Reference"))
vpcf_ax.legend(handles=legend_elements, loc="best")
vpcf_ax.get_xaxis().set_visible(False)
vpcf_ax.get_yaxis().set_visible(False)

# Save figure if desired; DO NOT close the plot window first, or an error will be thrown
savedir = Path(asksaveasfilename(defaultextension=".png"))
if savedir == Path("."):
    warn("Cannot save file to root directory; file save canceled!")
else:
    vpcf_fig.set_size_inches(10, 10)
    vpcf_fig.savefig(savedir, dpi=600, bbox_inches="tight")

# %% Plot distance map
lattice.get_vpcf_peak_params()
nn_dists = {}
for combo in combos:
    if combo[0] == combo[1]:
        n_peaks = 2
    else:
        n_peaks = 4
    nn_dists[combo] = lattice.plot_distances_from_vpcf_peak(f"{combo[0]}-{combo[1]}",
                                                            number_of_peaks_to_pick=n_peaks,
                                                            deviation_or_absolute="deviation",
                                                            return_nn_list=True)


# %% Generate nearest-neighbor distance histograms (per sublattice)
# TODO: Instead of finding the neighbors again, use the NN-distances list from the previous cell
bin_width: float = 0.01  # Angstroms
errorbar_offset: float = 50  # How far above the tallest histogram to place the error bars
print_stats: bool = True  # Print distribution statistics to stdout

if len(combos) > len(basic_colors):  # This is checked when plotting vPCFs, but in case that step is skipped check again
    raise RuntimeError(f"Trying to plot {len(combos)} histograms with {len(basic_colors)} colors: plot fewer vPCFs or"
                       " add additional colors.")
coord_dict = {lab: lattice.at_cols.loc[lattice.at_cols["elem"] == lab, ["x_fit", "y_fit", "u", "v"]].to_numpy()
              for lab in column_labels}
hist_dists = {combo: nn_distances(coord_dict[combo[0]],
                                  coord_dict[combo[1]],
                                  lattice.pixel_size_est) for combo in combos}
hist_stats = {combo: describe(dists) for combo, dists in hist_dists.items()}
if print_stats:
    for combo, stats in hist_stats.items():
        print(f"{combo} Distance Statistics:\n"
              f"Mean:     {stats.mean:0.3f}\n"
              f"Variance: {stats.variance:0.3f}\n"
              f"Skewness: {stats.skewness:0.3f}\n"
              f"Kurtosis: {stats.kurtosis:0.3f}")

max_counts = 0  # For setting the height of the errorbars
legend_handles = []
min_min, max_max = np.inf, 0  # For determining the correct xlim to include the whole data range
meanx = 0  # For determining where to center the xlim

hist_fig, hist_ax = plt.subplots()

# Plot the histograms
for combo, color in zip(combos, basic_colors):
    data = hist_dists[combo]
    counts, _, _ = hist_ax.hist(data, bins=np.arange(min(data), max(data)+bin_width, bin_width),
                                histtype="step", color=color)
    legend_handles.append(Patch(color=color, label=f"{combo[0]}–{combo[1]}"))
    # Keep track of xlim range
    if max(counts) > max_counts:
        max_counts = max(counts)
    if min(data) < min_min:
        min_min = min(data)
    if max(data) > max_max:
        max_max = max(data)

# Check for errorbar overlaps, and lift them up if they're overlapping
base_offset = max_counts + errorbar_offset
extra_offset = {combo: 0 for combo in combos}
needs_overlap_test = True
idx_map = {combo: i for i, combo in enumerate(combos)}
inv_idx_map = {i: combo for i, combo in enumerate(combos)}
while needs_overlap_test:
    needs_overlap_test = False  # This only gets set back to true if we find an overlap
    n_overlaps = [0] * len(combos)
    for a, b in combinations(combos, 2):  # Count the number of overlaps
        if extra_offset[a] == extra_offset[b]:  # Only check for overlaps if we're at the same vertical height
            a_mean, b_mean = hist_stats[a].mean, hist_stats[b].mean
            a_std, b_std = np.sqrt(hist_stats[a].variance), np.sqrt(hist_stats[b].variance)
            deltamean = abs(a_mean - b_mean)
            errsum = a_std + b_std
            if deltamean < errsum:  # The bars overlap
                needs_overlap_test = True  # Found an overlap, will need to check again after shifting
                n_overlaps[idx_map[a]] += 1
                n_overlaps[idx_map[b]] += 1
    lift_idx = np.argmax(n_overlaps)
    if not n_overlaps[lift_idx] == 0:
        extra_offset[inv_idx_map[lift_idx]] += errorbar_offset
errorbar_heights = [base_offset + extra for extra in extra_offset.values()]

for n, (combo, color, height, stats) in enumerate(zip(combos, basic_colors, errorbar_heights, hist_stats.values())):
    hist_ax.errorbar(x=stats.mean, y=height, xerr=np.sqrt(stats.variance),
                     fmt="x", color=color, capsize=plt.rcParams["lines.linewidth"]*2)
    meanx += stats.mean
meanx = meanx / len(combos)
x_spread = max(abs(meanx-min_min), abs(meanx-max_max))

hist_ax.legend(handles=legend_handles)
hist_ax.get_yaxis().set_ticks([])
hist_ax.set_xlim(meanx-x_spread, meanx+x_spread)
hist_ax.set_xlabel("Distance (\u212B)")

# Save figure if desired; DO NOT close the plot window first, or an error will be thrown
savedir = Path(asksaveasfilename(defaultextension=".png"))
if savedir == Path("."):
    warn("Cannot save file to root directory; file save canceled!")
else:
    hist_fig.set_size_inches(10, 6)
    hist_fig.savefig(savedir, dpi=600, bbox_inches="tight")

#%%
