from tkinter.filedialog import askopenfilenames, askopenfilename
from pathlib import Path
import numpy as np
import scipy
from numpy.typing import ArrayLike
import SingleOrigin as so
import matplotlib
import matplotlib.pyplot as plt
import scipy.spatial.distance as scidist

matplotlib.use("TkAgg")
plt.style.use('dark_background')


def xy_to_rt(xy: tuple) -> tuple:
    """Convert Cartesian to polar coordinates."""
    r = np.linalg.norm(xy)
    if xy[0] == 0:
        if xy[1] > 0:
            t = np.pi/2
        else:
            t = -1*np.pi/2
    else:  # Non-zero x avoids divide-by-zero
        t = np.arctan(xy[1]/xy[0])
        if xy[0] < 0:  # Quadrants II and III
            t += np.pi
    return r, t


def rt_to_xy(rt: tuple) -> tuple:
    """Convert polar to Cartesian coordinates."""
    x = rt[0]*np.cos(rt[1])
    y = rt[0]*np.sin(rt[1])
    return x, y


def similarity_transform(points: list[tuple[float, float]] | ArrayLike,
                         scale_factor: float = 1,
                         rotation: float = 0) -> list[tuple[float, float]]:
    """A similarity transform is an affine transform with no shear component, only scaling and rotation.

    Args:
        points: A collection of points to transform, in polar (r, t) form.
        scale_factor: The (multiplicative) radial scaling factor.  Must be positive.
        rotation: The (additive) rotation factor.  Must be between 2pi and -2pi.
    """
    if rotation > 2*np.pi or rotation < -2*np.pi:
        raise RuntimeError("Rotation > full revolution")
    if scale_factor < 0:
        raise RuntimeError("Negative scale factor")
    return [(point[0]*scale_factor, point[1]+rotation) for point in points]


def loss(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Loss function for optimization fitting (compare distances in two point clouds)"""
    # Lists should be sorted so that permutations of point order do not affect the loss function
    # TODO: If this loss function doesn't work, try constructing kd trees instead
    _a = np.array(sorted(sorted(a, key=lambda x: x[1]), key=lambda x: x[0]))
    _b = np.array(sorted(sorted(b, key=lambda x: x[1]), key=lambda x: x[0]))
    dist_matrix = scipy.spatial.distance.cdist(_a, _b, metric="sqeuclidean")  # sqeuclidean might be faster?
    return np.trace(dist_matrix)


# %% Open .cif files for each relevant phase
file_list = [Path(file) for file in
             askopenfilenames(filetypes=[("Crystallographic Information File", ".cif")])]
print([f"{i}: {f.stem}" for i, f in enumerate(file_list)])
ignore_elements: list[str] = ["O"]  # The listed elements will be skipped when generating vPCF coordinates

# %% Generate vPCF coordinates for the perfect structures
combine_tol: float = 0.5  # Tolerance for combining close vPCF peaks into a single peak
order: int = 1  # The highest order spots to guarantee the generation of
# Tuples are (zone axis vector, basis vector 2, basis vector 3)
zones = [((0, 0, 1), (5, 0, 0), (0, 5, 0)),
         ((0, 1, 0), (5, 0, 0), (0, 0, -5)),
         ((1, 0, 0), (0, 0, -5), (0, 5, 0)),
         ((1, 1, 0), (5, -5, 0), (0, 0, -5)),
         ((1, 0, 1), (5, 0, -5), (0, -5, 0)),
         ((0, 1, 1), (0, -5, 5), (5, 0, 0)),
         ((1, 1, 1), (5, 0, -5), (-5, 10, -5))]

structural_mapping = {(fname, axes): None for fname in file_list for axes in zones}  # Initialize empty
for mapping in structural_mapping.keys():
    cif_path = mapping[0]
    za, a1, a2 = mapping[1]
    uc = so.UnitCell(str(cif_path))  # Must cast path object to string for SingleOrigin to handle it
    uc.project_zone_axis(za, a1, a2,
                         ignore_elements=ignore_elements,
                         reduce_proj_cell=False)
    uc.combine_prox_cols(toler=combine_tol)

    v_pcf, origin = so.v_pcf(xlim=(-6, 6), ylim=(-6, 6), d=0.01,
                             coords1=uc.at_cols.loc[uc.at_cols["elem"] == "Hf", ["x", "y"]].to_numpy())
    peaks = np.argwhere(so.detect_peaks(v_pcf, min_dist=4))
    peak_origin_coordinates = [((peak[0]-origin[0])*0.01, (peak[1]-origin[1])*0.01) for peak in peaks]

    # Combine close vPCF points (which would be indistinguishable in the image)
    dists = scidist.squareform(scidist.pdist(np.array(peak_origin_coordinates),
                                             metric="sqeuclidean"))
    edges = set()
    for i in range(len(peak_origin_coordinates)):
        neighbors = np.argwhere(dists[i, :] < combine_tol).flatten()
        if neighbors.size == 1:  # Disregard points which have no close neighbors
            continue
        new_edges = {frozenset([i, n]) for n in neighbors if n != i}
        edges.update(new_edges)

    # neighbor_mapping = {}
    # for i in range(len(peak_origin_coordinates)):
    #     neighbors = np.argwhere(dists[i, :] < combine_tol).flatten()
    #     neighbors = np.delete(neighbors, np.argwhere(neighbors == i))  # Self-neighboring doesn't count
    #     if neighbors.size == 0:
    #         continue
    #     neighbor_mapping[i] = neighbors

    peak_polar = [xy_to_rt(xy) for xy in peak_origin_coordinates]
    structural_mapping[mapping] = peak_polar

# %% Test cell (only runs if test is True, so you can run the whole script without this popping anything up)
test: bool = True  # Set this to enable displaying coordinate plots (for debugging)
if test:  # Plot one of the phase-and-zones to check if things are working
    for i in range(len(structural_mapping)):
        test_polar = list(structural_mapping.values())[i]  # Select phase-and-zone

        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        ax.set_rlim(rmin=0, rmax=np.ceil(np.max(np.array(list(test_polar))[:, 0])))
        ax.set_thetagrids((0, 90, 180, 270), ("0", u"\u03c0/2", u"\u03c0", "3"+u"\u03c0/2"))
        ax.scatter(np.array(list(test_polar))[:, 1], np.array(list(test_polar))[:, 0])
        fig.show()

# %% SingleOrigin setup
fitting_file: Path = file_list[2]  # Change to select the .cif file to use for initial fitting

uc = so.UnitCell(str(fitting_file), origin_shift=(0.1, 0.1, 0))
uc.project_zone_axis((0, 1, 1),  # Zone axis direction
                     (0, 2, 0),
                     (2, 0, 0),
                     ignore_elements=ignore_elements,
                     reduce_proj_cell=False)
uc.combine_prox_cols(toler=combine_tol)

image, _ = so.load_image(path=askopenfilename(), display_image=False, images_from_stack="all")
hrimage = so.HRImage(image)
lattice = hrimage.add_lattice("lattice", uc)
lattice.get_roi_mask_polygon()

# %% Fit lattice to image, and refine
lattice.fft_get_basis_vect(a1_order=4, a2_order=4, sigma=3)
lattice.define_reference_lattice(plot_ref_lattice=False)
lattice.fit_atom_columns(buffer=10, local_thresh_factor=0.0, parallelize=True,
                         use_circ_gauss=True, watershed_line=True)
plot_fit_lattice: bool = False
if plot_fit_lattice:
    hrimage.plot_atom_column_positions(fit_or_ref='fit', outlier_disp_cutoff=100, scalebar_len_nm=None)

# %%
# TODO: Copy in Sebastian's code to get an experimental vPCF to compare against
