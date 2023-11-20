from pathlib import Path
from tkinter.filedialog import askopenfilename, asksaveasfilename
from itertools import combinations, combinations_with_replacement
from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import quickcrop as qc
import SingleOrigin as so
import matplotlib
from tifffile import imread
import hyperspy.api as hs
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from data_analysis_scripts.cse_secrets import MP_API_KEY

matplotlib.use("TkAgg")


def _minmax_norm(img):
    _min, _max = np.min(img), np.max(img)
    normed = (img - _min) / (_max - _min)
    return normed


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

fixed_path = Path("E:/Users/Charles/AlN.cif")  # Local path to .cif file
mp_id = "mp-661"  # Materials Project ID for desired structure, including the leading "mp-"

origin_shift: tuple[float, float, float] = (0, 0, 0)  # Set to shift the unit cell origin of the structure on load
# To relabel atomic sites in the unit cell on load, set as a dict from default label to new label
# If there are no sites to relabel, leave the dictionary empty
# This is mainly cosmetic (affect label in vPCF plots)
replacement_labels: dict[str, str] = {"Al": "Al/Sc"}

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

uc.project_zone_axis((1, 1, 0),  # Zone axis direction
                     (1, -1, 0),  # Apparent horizontal axis in projection
                     (0, 0, 1),  # Most vertical axis in projection
                     ignore_elements=[]  # List of elements to ignore (e.g. light elements in a HAADF image)
                     )
uc.combine_prox_cols(toler=1e-2)  # 1e-2 works well as a tolerance, but adjust as needed

if plot_projected_cell:
    uc.plot_unit_cell()

# %% Create the SingleOrigin HRImage object
hr_img = so.HRImage(img)
lattice = hr_img.add_lattice("(Al,Sc)N",  # Human-readable name for the lattice in the image
                             uc,
                             origin_atom_column=1)  # Index in uc of column to use as fitting origin, None == default

# %% Reciprocal-space rough fit of lattice
# Both of these methods spawn windows that must be interacted with before they time out!
# Rerun the cell if you accidentally let them time out
# If some FFT peaks are weak or absent (such as forbidden reflections), specify the order of the first peak that
# is clearly visible
lattice.fft_get_basis_vect(a1_order=1,  # Order of peak corresponding to planes in the most horizontal direction
                           a2_order=2)  # Order of peak corresponding to planes in the most vertical direction
lattice.define_reference_lattice()

# %% Fit the atom columns
lattice.fit_atom_columns(buffer=10,  # Pixel buffer around image edges to avoid partial columns
                         local_thresh_factor=0.0,  # Watershed segmentation threshold
                         use_background_param=True,  # Fit the background intensity
                         use_bounds=True,  # Bound allowed column fit position
                         use_circ_gauss=True,  # Force gaussians to be circular (major axis == minor axis)
                         parallelize=True,  # Parallelize the computation
                         peak_grouping_filter="auto",  # Apply a filter for simultaneous column fitting
                         peak_sharpening_filter="auto")  # Apply a filter to improve column position fitting

# Must have only one column per projected unit cell.  If no sublattice meets this criteria, specify a specific column
# in the projected cell.
lattice.refine_reference_lattice(filter_by="elem",  # Which dataframe column to select on
                                 sites_to_use="Al/Sc")  # Which entry to select

# %% Assess fit quality
# All of these steps are optional but can help with assessing a failed fit; uncomment desired lines
# See function documentation for param descriptions
# lattice.show_masks()

# hr_img.plot_atom_column_positions(scatter_kwargs_dict={"s": 20},
#                                   scalebar_len_nm=None,
#                                   fit_or_ref="fit")

# _ = lattice.get_fitting_residuals()

# plot_disp_vecs also prints out displacement statistics
# hr_img.plot_disp_vects(sites_to_plot=["N"],
#                        arrow_scale_factor=2)

# %% Generate the relevant vCPFs
column_labels: set = {"N", "Al/Sc"}  # Which columns to generate vPCFs for (usually element names)
self_vpcf: bool = True  # True to generate an (E | E) vPCF for each element E
plot_ref: bool = True  # True to add points onto the vPCF plot from the (unrefined) reference lattice
xlimits, ylimits = (-40, 40), (-60, 60)  # Limits for vPCF plotting (in the units of the fitted column positions?)

if self_vpcf:
    combos = sorted(list(combinations_with_replacement(column_labels, 2)))
else:
    combos = sorted(list(combinations(column_labels, 2)))

coord_dict = {lab: lattice.at_cols.loc[lattice.at_cols["elem"] == lab, "x_fit":"y_fit"].to_numpy()
              for lab in column_labels}
vpcfs = {f"({combo[0]} | {combo[1]})": so.v_pcf(coords1=coord_dict[combo[0]],
                                                coords2=coord_dict[combo[1]],
                                                xlim=xlimits, ylim=ylimits) for combo in combos}
if plot_ref:  # Generate the reference vPCFs if needed
    ref_coord_dict = {lab: lattice.at_cols.loc[lattice.at_cols["elem"] == lab, "x_ref":"y_ref"].to_numpy()
                      for lab in column_labels}
    ref_vpcfs = {f"({combo[0]} | {combo[1]}) (ref)": so.v_pcf(coords1=ref_coord_dict[combo[0]],
                                                              coords2=ref_coord_dict[combo[1]],
                                                              xlim=xlimits, ylim=ylimits) for combo in combos}
_arbitrary_vpcf = vpcfs[next(iter(vpcfs))]  # The origin should be the same for all vPCFs, so just grab one of them
origin_x, origin_y = _arbitrary_vpcf[1][0], _arbitrary_vpcf[1][1]

# %% Plotting setup for the vPFCs
# These colors come from the IBM colorblind palette and should have okay contrast even in grayscale
basic_colors = {"green":   "#117733ff",
                "magenta": "#dc267fff",
                "orange":  "#fe6100ff"}
# Three colors should be sufficient for two elements with self-vPCFs enabled, or three without; more vPCFs on one
# plot is probably a bad idea but if you want to do that you'll need more colors
if len(combos) > len(basic_colors):
    raise RuntimeError(f"Trying to plot {len(combos)} vPCFs with {len(basic_colors)} colors: plot fewer vPCFs or add"
                       " additional colors.")

cmaps = [LinearSegmentedColormap.from_list(f"black-{cname}-white",
                                           ["#00000000", cval, "#ffffffff"], 256)
         for cname, cval in basic_colors.items()]

# %% Plot the vPCFs
# Adjust min and max values to get the desired level of color saturation on the vPCFs
minval: int = 0
maxval: int = 200

fig = plt.figure()
ax = plt.axes()
ax.set_facecolor("black")
for vpcf, cmap in zip(vpcfs.values(), cmaps):
    ax.imshow(vpcf[0], cmap=cmap, vmin=minval, vmax=maxval, interpolation=None)
if plot_ref:
    # noinspection PyUnboundLocalVariable
    for ref_vpcf in ref_vpcfs.values():
        ax.imshow(ref_vpcf[0], cmap=LinearSegmentedColormap.from_list("transparent-blue",
                                                                      ["#00000000", "#22aaffff"], 2),
                  vmax=1, vmin=0, interpolation="gaussian")
ax.plot(origin_x, origin_y, "wx")  # Plot the origin as an "x"
ax.set_title("Vector Partial Pair Correlation Functions for "
             f"{' & '.join(filter(None, [', '.join(list(column_labels)[:-1]), list(column_labels)[-1]]))}")
legend_elements = [Patch(facecolor=c, edgecolor=c, label=lab) for c, lab in zip(basic_colors, vpcfs.keys())]
legend_elements.append(Patch(facecolor="#22aaffff", edgecolor="#22aaffff", label="Reference"))
ax.legend(handles=legend_elements, loc="best")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.show()

# %% Save figure if desired
savedir = Path(asksaveasfilename(defaultextension=".png"))
if savedir == Path("."):
    raise RuntimeError("Cannot save file to root directory; file save canceled!")
else:
    fig.savefig(savedir, dpi=300, bbox_inches="tight")

# %% END
