import copy
from pathlib import Path
from typing import Literal
from utils import tk_popover

from tifffile import imread
from warnings import warn

from itertools import combinations
import numpy as np
import SingleOrigin as so
from scipy.stats import describe, gaussian_kde, shapiro
from scipy.spatial import Voronoi
import pandas as pd
from libpysal import weights
from esda import Moran, Moran_Local

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
matplotlib.use("TkAgg")
plt.style.use('dark_background')

########################
#        PART 0        #
# Function Definitions #
########################


def minmax_norm(img):
    """Normalize a 2D array into the range [0, 1]"""
    # Underscores avoid overwriting reserved names
    _min, _max = np.min(img), np.max(img)
    return (img - _min) / (_max - _min)


def get_uv_neighbors(row: pd.Series | pd.DataFrame,
                     df: pd.DataFrame,
                     uv_offsets: list[tuple[float, float]])\
        -> list[int] | None:
    """Apply to dataframe to add column of neighbor indices for each site
    Args:
        row: Current row for df.apply
        df: Dataframe for df.apply (should be the same frame row comes from)
        uv_offsets: A list of (u, v) cooridnate offsets for which columns should count as a neighbor. If a particular
          (u, v) coordinate does not point to an atom column, it is skipped. Common patterns include:
            - Sqaure rook: [(1, 0), (-1, 0), (0, 1), (0, -1)]
            - Sqaure bishop: [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            - Square queen: rook.extend(bishop)
            - Wurtzite 11-20: [(2/3, 1/2), (2/3, -1/2), (1/3, 1/2), (1/3, -1/2), (0, 1), (0, -1),
              (-1/3, 1/2), (-1/3, -1/2), (-2/3, 1/2), (-2/3, -1/2)]
    Returns:
        List of neighbors, as indexed in df
    """
    u, v = row["u"], row["v"]
    neighborhood = set()  # Avoid duplicates due to repeated (u, v) offsets (e.g. [(0, 1), (0, 1)])
    for u_off, v_off in uv_offsets:
        match = df.loc[(np.isclose(df["u"], u+u_off)) & (np.isclose(df["v"], v+v_off))]
        if len(match) != 1:
            # We should never get more than one match, but if we do we'll skip them since something is wrong
            # We might get 0 matches for two reasons: either we're at an image edge, or else the (u, v) offset
            #  doesn't point to a valid site from this one (happens when there's more than one kind of site
            #  per projected cell)
            continue
        neighborhood.add(match.index[0])

    # Sometimes we find no neighbors if the column was fit poorly; just throw it out
    if not neighborhood:
        neighborhood = None
    return list(neighborhood)


##################
#     PART 1     #
# Import and Fit #
##################
# %% Image import
path = Path(tk_popover())
match path.suffix:
    case ".tif":
        # noinspection PyTypeChecker
        img = minmax_norm(np.array(imread(path)))
    case ".emd":
        import hyperspy.api
        from importlib.metadata import version
        if version("hyperspy") > "2.0rc0":
            warn("File IO was deprecated in HyperSpy versions 2.0rc0 and beyond; .emd file importing will not work.")
        infile = hyperspy.api.load(path)
        # The only kind of data it makes sense to load from an .emd is from the HAADF detector
        # iDPC / dDPC images are not computed and saved in the .emd file
        img = minmax_norm(next(signal for signal in infile
                               if signal.metadata.General.title == "HAADF"))
    case "":
        raise RuntimeError("No type extension in file name; either file has no extension or no file was selected.")
    case _:
        raise RuntimeError(f"Unsupported filetype: {path.suffix}")

# %% Import the .cif structure file
load_method: Literal[
    "fixed",  # Load via a fixed file path, given below, without a file picker window
    "interactive",  # Interactively select the file using a (native) file picker window
    "mp_api"  # Load a cif from Materials Project; requires an API key!
    ] = "fixed"
fixed_path = Path("E:/Users/Charles/AlN.cif")  # Local path to .cif file
mp_id = "mp-661"  # Materials Project ID for desired structure, including the leading "mp-"

origin_shift: tuple[float, float, float] = (0, 0, 0)  # Set to shift the unit cell origin of the structure on load
# To relabel atomic sites in the unit cell on load, set as a dict from default label to new label
# If there are no sites to relabel, leave the dictionary empty
# This is mainly cosmetic (affect label in vPCF plots)
replacement_labels: dict[str, str] = {"Al": "Al/Gd"}

match load_method:
    case "fixed":
        uc = so.UnitCell(str(fixed_path), origin_shift=origin_shift)
    case "interactive":
        interactive_path = Path(tk_popover())
        uc = so.UnitCell(str(interactive_path), origin_shift=origin_shift)
    case "mp_api":
        print(f"Searching for {mp_id}...")
        import tempfile
        from mp_api.client import MPRester
        from pymatgen.io.cif import CifWriter
        from data_analysis_scripts.cse_secrets import MP_API_KEY
        mp_api_key = MP_API_KEY  # Personal API key (tied to MP account)
        with MPRester(mp_api_key) as mp:
            mp_struct = mp.get_structure_by_material_id(mp_id)
            # We must write to a temporary .cif file for SingleOrigin to be able to read the structure
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file_path = Path(tmp_dir, "tmp.cif")
                writer = CifWriter(mp_struct, symprec=True, refine_struct=True)
                writer.write_file(tmp_file_path)
                uc = so.UnitCell(str(tmp_file_path), origin_shift=origin_shift)
    case _:
        raise NotImplementedError(f"Unrecognized file load method: {load_method}")

for key, replacement in replacement_labels.items():
    uc.atoms.replace(key, replacement, inplace=True)

# %% Project the unit cell along a given zone axis
plot_projected_cell: bool = True  # Set True to pop up a visualization of the projected unit cell for verification

uc.project_zone_axis((1, 1, 0),  # Zone axis direction
                     (1, -1, 0),  # Apparent horizontal axis in projection
                     (0, 0, 1),  # Most vertical axis in projection
                     ignore_elements=["N"],  # List of elements to ignore (e.g. light elements in a HAADF image)
                     reduce_proj_cell=True)
uc.combine_prox_cols(toler=1e-2)  # 1e-2 works well as a tolerance, but adjust as needed

if plot_projected_cell:
    uc.plot_unit_cell()

# %% Create the SingleOrigin HRImage object
latt_dict_name: str = "AlN"  # Human-readable name for the lattice in the image
origin_ac: int | None = None  # Index in uc of column to use as fitting origin, default is None
px_size: float | None = 8.7  # Image pixel size (pm); if None, will be estimated from the fitted lattice
crop: None | Literal[  # Set to None to fit entire field of view
    "quick",  # Quick interactive cropping (square only)
    "flexible",  # Select an arbitrary contiguous polygonal region
    ] = "flexible"

match crop:
    case None:
        hr_img = so.HRImage(img, pixel_size_cal=px_size)
        # noinspection PyTypeChecker
        lattice = hr_img.add_lattice(latt_dict_name, uc, origin_atom_column=None)
    case "quick":
        import quickcrop
        img = quickcrop.gui_crop(img)
        hr_img = so.HRImage(img, pixel_size_cal=px_size)
        # noinspection PyTypeChecker
        lattice = hr_img.add_lattice(latt_dict_name, uc, origin_atom_column=None)
    case "flexible":
        hr_img = so.HRImage(img, pixel_size_cal=px_size)
        # noinspection PyTypeChecker
        lattice = hr_img.add_lattice(latt_dict_name, uc, origin_atom_column=None)
        lattice.get_roi_mask_polygon()
    case _:
        raise NotImplementedError("Unrecognized cropping type")

# %% Reciprocal-space rough fit of lattice

# Both of these methods spawn windows that must be interacted with before they time out!
# Rerun the cell if you accidentally let them time out
# If some FFT peaks are weak or absent (such as forbidden reflections), specify the order of the first peak that
# is clearly visible
lattice.fft_get_basis_vect(a1_order=1,  # Order of peak corresponding to planes in the most horizontal direction
                           a2_order=2,  # Order of peak corresponding to planes in the most vertical direction
                           thresh_factor=0.5)  # Decrease to detect more peaks
lattice.define_reference_lattice()

# %% Fit the atom columns
lattice.fit_atom_columns(buffer=10,  # Pixel buffer around image edges to avoid partial columns
                         bkgd_thresh_factor=0,  # Watershed segmentation threshold
                         use_bounds=True,  # Bound allowed column fit position
                         pos_bound_dist=0.3,  # Position fit shift limit (Angstrom)
                         use_circ_gauss=False,  # Force gaussians to be circular (major axis == minor axis)
                         parallelize=True,  # Parallelize the computation
                         peak_grouping_filter=None,  # Apply a filter for simultaneous column fitting
                         peak_sharpening_filter="auto")  # Apply a filter to improve column position fitting

# Must have only one column per projected unit cell.  If no sublattice meets this criteria, specify a specific column
# in the projected cell.
lattice.refine_reference_lattice(filter_by="elem",  # Which dataframe column to select on
                                 sites_to_use="Al/Gd")  # Which entry to select

# %% Assess fit quality
# All of these steps are optional but can help with assessing a failed fit; uncomment desired lines
# It's a good idea to at least assess_positions to make sure fitting went as expected
# See function documentation for param descriptions
assess_masks: bool = False
assess_positions: bool = True
assess_residuals: bool = False
assess_displacement: bool = False

if assess_masks:
    lattice.show_masks()
if assess_positions:
    hr_img.plot_atom_column_positions(scatter_kwargs_dict={"s": 20}, scalebar_len_nm=None, fit_or_ref="fit")
if assess_residuals:
    _ = lattice.get_fitting_residuals()
if assess_displacement:
    hr_img.plot_disp_vects(sites_to_plot=["Al/Gd"], arrow_scale_factor=2)

###################
#     PART 2      #
# vPCF Generation #
###################
# %% Setup vPCFs
hr_img_rot = hr_img.rotate_image_and_data("AlN", "a1", "up")
lattice = hr_img_rot[0].latt_dict["AlN_rot"]
pair_pair: bool = True  # If true, will get vPCFs within _and_ between sublattices, otherwise only within
pxsize: float = 0.01  # Angstrom
column_labels: set = {"Al/Gd"}  # Which columns to generate vPCFs for (usually element names)
xlimits, ylimits = (-1.1, 1.1), (-1.1, 1.1)  # Limits for vPCF plotting (unit cells)
# These colors come from the IBM colorblind palette and should have okay contrast even in grayscale
basic_colors = {"green":   "#117733ff",
                "magenta": "#dc267fff",
                "orange":  "#fe6100ff"}

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
maxval: int = 100

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
    ref_vpcfs = {combo: so.get_vpcf(coords1=ref_coord_dict[combo[0]],
                                    coords2=ref_coord_dict[combo[1]],
                                    xlim=ref_xlimits, ylim=ref_ylimits, d=pxsize)
                 for combo in combos}

vpcf_fig, vpcf_ax = plt.subplots()
for vpcf, cmap in zip([value for key, value in lattice.vpcfs.items() if key != "metadata"], cmaps):
    vpcf_ax.imshow(vpcf, cmap=cmap, vmin=minval, vmax=maxval, interpolation=None)
if plot_ref:
    # noinspection PyUnboundLocalVariable
    for ref_vpcf in ref_vpcfs.values():
        vpcf_ax.imshow(ref_vpcf[0], cmap=LinearSegmentedColormap.from_list("transparent-orange",
                                                                           ["#00000000", "#fe6100ff"], 2),
                       vmax=1, vmin=0, interpolation="gaussian")

vpcf_ax.plot(origin_x, origin_y, "wx")  # Plot the origin as an "x"
vpcf_ax.set_title("Vector Partial Pair Correlation Functions for "
                  f"{' & '.join(filter(None, [', '.join(list(column_labels)[:-1]), list(column_labels)[-1]]))}")
legend_elements = [Patch(facecolor=c, edgecolor=c, label=f"{combo[0]}–{combo[1]}")
                   for c, combo in zip(basic_colors, combos)]
legend_elements.append(Patch(facecolor="#fe6100", edgecolor="#fe6100", label="Reference"))
vpcf_ax.legend(handles=legend_elements, loc="best")
vpcf_ax.get_xaxis().set_visible(False)
vpcf_ax.get_yaxis().set_visible(False)

# Save figure if desired; DO NOT close the plot window first, or an error will be thrown
savedir = Path(tk_popover(save=True, defaultextension=".png"))
if savedir == Path("."):
    warn("Cannot save file to root directory; file save canceled!")
else:
    vpcf_fig.set_size_inches(10, 10)
    vpcf_fig.savefig(savedir, dpi=600, bbox_inches="tight")


# %% Plot distance map
n_peaks = 1  # number of peaks to fit
lattice.get_vpcf_peak_params()
nn_dists = {}
for combo in combos:
    nn_dists[combo] = lattice.plot_distances_from_vpcf_peak(f"{combo[0]}-{combo[1]}",
                                                            number_of_peaks_to_pick=n_peaks,
                                                            deviation_or_absolute="deviation",
                                                            return_nn_list=True)

#################
#    PART 3     #
# Global Survey #
#################
# %% Generate a histogram of column intensities
try:
    frame = copy.deepcopy(hr_img.latt_dict[latt_dict_name].at_cols)
except KeyError:
    # Lattice may have been rotated
    frame = copy.deepcopy(hr_img.latt_dict[latt_dict_name+"_rot"].at_cols)
frame.drop(["site_frac", "x", "y", "weight"], axis=1, inplace=True)  # We don't need these cols
frame.reset_index(drop=True, inplace=True)
ints = list(frame["total_col_int"])
ints = minmax_norm(np.array(ints))

dist_params = describe(ints)
outlier_thresh = (max(0, dist_params.mean - 4*np.sqrt(dist_params.variance)),
                  min(1, dist_params.mean + 4*np.sqrt(dist_params.variance)))
ints = [i for i in ints if outlier_thresh[0] < i < outlier_thresh[1]]  # Filter outliers
ints = minmax_norm(np.array(ints))  # Re-norm w/o outliers
dist_params = describe(ints)  # Re-calculate stats

int_kernel = gaussian_kde(ints)
density_estimate = int_kernel.evaluate(np.linspace(0, 1, 1000))

fig, hax = plt.subplots()
kdeax = hax.twinx()
hax.hist(ints, bins="auto", histtype="step", color="#fe6100")
hax.set_ylabel("Counts")
hax.set_xlabel("Normalized Intensity")
kdeax.plot(np.linspace(0, 1, 1000), density_estimate, "-", c="#785ef0")
kdeax.set_yticks([])
kdeax.set_ylim(bottom=0)
hax.set_xlim(0, 1)
hax.spines[["right", "top"]].set_visible(False)
kdeax.spines[["right", "top"]].set_visible(False)

# noinspection PyUnresolvedReferences
statstr = (f"Mean: {dist_params.mean:.3g}\n"
           f"St. Deviation: {np.sqrt(dist_params.variance):.3g}\n"
           f"Skewness: {dist_params.skewness:.3g}\n"
           f"Excess Kurtosis: {dist_params.kurtosis:.3g}\n"
           f"Data is {'not' if shapiro(ints)[1]<0.05 else ''} normally distributed")
hax.text(1, 1, statstr, transform=hax.transAxes, verticalalignment="top", horizontalalignment="right",
         color="white", bbox={"boxstyle": "round", "facecolor": "black"})
plt.show()

###############
#   PART 4    #
# Local Study #
###############
#%% Regenerate frame, in case we need it or it's been mucked with, setup for spatial stats
n_sigmas: float = 2  # Number of standard deviations away from the mean a point can be before outlier rejection

try:
    frame = copy.deepcopy(hr_img.latt_dict[latt_dict_name].at_cols)
except KeyError:
    # Lattice may have been rotated
    frame = copy.deepcopy(hr_img.latt_dict[latt_dict_name+"_rot"].at_cols)
frame.drop(["site_frac", "x", "y", "weight", "elem"], axis=1, inplace=True)  # We don't need these cols
frame.reset_index(drop=True, inplace=True)

uv_offsets = [(2/3, 1/2), (2/3, -1/2), (1/3, 1/2), (1/3, -1/2), (0, 1),
              (0, -1), (-1/3, 1/2), (-1/3, -1/2), (-2/3, 1/2), (-2/3, -1/2)]
frame["neighborhood"] = frame.apply(lambda row: get_uv_neighbors(row, frame, uv_offsets), axis=1)

# Filter outlier and create libpysal weights matrix object
mu, sig = frame["total_col_int"].mean(), frame["total_col_int"].std()
non_outliers = frame[(frame["total_col_int"] >= mu-n_sigmas*sig) & (frame["total_col_int"] <= mu+n_sigmas*sig)].index
filtered_frame = frame.loc[non_outliers].copy()
filtered_frame["neighborhood"] = filtered_frame["neighborhood"].apply(lambda neighbors: [n for n in neighbors
                                                                                         if n in non_outliers])
adjmap = {}
for i, row in filtered_frame.iterrows():
    adjmap[str(i)] = tuple(map(str, row.neighborhood))
w = weights.W(adjmap, silence_warnings=True)
w.transform = "r"  # Row-standard form

# %% Get Moran's I stats (global, local)
alpha: float = 0.05  # Statistical power for false discovery rate control

ints = [x for x in filtered_frame["total_col_int"]]

global_stats = Moran(ints, w, permutations=10000)
print(f"Global Moran's I:     {round(global_stats.I, 5)}\n"
      f"Expected I for CSR:  {round(global_stats.EI_sim, 5)}\n"
      f"p-value (10k perms.): {round(global_stats.p_sim, 5)}")
if global_stats.p_sim < 0.05:
    print("Observed distribution not consistent with CSR")
else:
    print("Observed distribution consistent with CSR")

local_stats = Moran_Local(ints, w, permutations=10000)
# Add local stats back to frame
filtered_frame["moran_I"] = local_stats.Is
filtered_frame["moran_quadrant"] = local_stats.q
filtered_frame["moran_p"] = local_stats.p_sim

# FDR Control
ps = {i: p for i, p in enumerate(local_stats.p_sim)}
ps = {k: v for k, v in sorted(ps.items(), key=lambda item: item[1])}
largest = 0
for i, p in enumerate(ps.values()):  # Find largest i s.t. p_i <= i/len(ps)*alpha
    if p <= i/len(ps)*alpha:
        largest = i
fdr_keep = {}
for i, k in enumerate(ps.keys()):  # Mark locations which pass the FDR criteria
    if i <= largest:
        fdr_keep[k] = 1
    else:
        fdr_keep[k] = 0
filtered_frame["fdr_keep"] = list(fdr_keep.values())  # Update frame

# %% Plot local Moran data
# noinspection PyTypeChecker
vor = Voronoi([(x, y) for x, y in zip(filtered_frame["x_fit"], filtered_frame["y_fit"])])
members = {i for i, (keep, quad) in enumerate(zip(filtered_frame["fdr_keep"], filtered_frame["moran_quadrant"]))
           if keep == 1 and quad in [1, 3]}

members_tmp = copy.copy(members)
for m in members_tmp:
    members.update(filtered_frame.iloc[m]["neighborhood"])  # Expand from core members to include their neighbors
del members_tmp

norm = Normalize(vmin=min(ints), vmax=max(ints), clip=True)
mapper = ScalarMappable(norm=norm, cmap="inferno")

fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 4, width_ratios=[0.02, 0.05, 1, 1], height_ratios=[0.02, 1], figure=fig)
axs = [fig.add_subplot(gs[1, 1]),
       fig.add_subplot(gs[1, 2]),
       fig.add_subplot(gs[1, 3])]
for ax in [axs[1], axs[-1]]:  # Formatting applied to both axes
    ax.imshow(img, cmap="bone")
    ax.axis("off")
    ax.set_xlim(left=0, right=img.shape[0])
    ax.set_ylim(bottom=img.shape[1], top=0)
axs[1].set_title("Column Intensity")
axs[-1].set_title("Cluster Members")
cbar = fig.colorbar(mapper, cax=axs[0])
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.yaxis.set_label_position("left")

for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if -1 not in region:
        polygon = [vor.vertices[i] for i in region]
        axs[1].fill(*zip(*polygon), color=mapper.to_rgba(filtered_frame.iloc[r]["total_col_int"]), alpha=0.5)
        if r in members:
            if filtered_frame.iloc[r]["moran_quadrant"] == 1:  # HH cluster core
                c = "goldenrod"
                a = 0.5
            elif filtered_frame.iloc[r]["moran_quadrant"] == 3:  # LL cluster core
                c = "maroon"
                a = 0.5
            else:  # Non-core cluster member
                c = "white"
                a = 0.0
            axs[-1].fill(*zip(*polygon), color=c, alpha=a)
plt.show()


#%% Generate nearest-neighbor distance histograms
# TODO: THE BELOW CODE IS BROKEN!
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
savedir = Path(tk_popover(save=True, defaultextension=".png"))
if savedir == Path("."):
    warn("Cannot save file to root directory; file save canceled!")
else:
    hist_fig.set_size_inches(10, 6)
    hist_fig.savefig(savedir, dpi=600, bbox_inches="tight")

#%%
