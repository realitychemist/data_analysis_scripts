import os
import sys
import matplotlib
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from copy import copy
from atomai import utils
from pathlib import Path
from warnings import warn
from sklearn import cluster
from tkinter import messagebox
from scipy import optimize, spatial
from contextlib import contextmanager
from scipy.spatial.distance import squareform, pdist
from tkinter.filedialog import askopenfilenames, askopenfilename
from SingleOrigin import get_vpcf, load_image, HRImage, UnitCell, detect_peaks
from networkx import draw_networkx, connected_components, from_edgelist

matplotlib.use("TkAgg")  # Needed to get interactive plots in PyCharm
plt.style.use("dark_background")
plt.rcParams.update({"figure.max_open_warning": 0})  # Suppress warnings when testing vPCF output


def xy_to_rt(xy: tuple[float, float]) -> tuple[float, float]:
    """Convert a single point from Cartesian (x, y) to polar (r, t) coordinates."""
    r = np.linalg.norm(xy)
    if xy[0] == 0:  # Avoid the divide-by-zero inside arctan
        if xy[1] >= 0:
            t = np.pi/2
        else:
            t = 3*np.pi/2
    else:
        t = np.arctan(xy[1] / xy[0])
        if xy[0] < 0:  # Quadrants II and III
            t += np.pi
    if t < 0:
        t = 2*np.pi + t
    return r, t


def rt_to_xy(rt: tuple[float, float]) -> tuple[float, float]:
    """Convert a single point from polar (r, t) to Cartesian (x, y) coordinates."""
    x = float(rt[0] * np.cos(rt[1]))
    y = float(rt[0] * np.sin(rt[1]))
    return x, y


def similarity_transform(polar_pt: tuple[float, float],
                         r_scale: float, rot: float) -> tuple[float, float]:
    """Similarity (rotation & scaling) transform for a single point, in polar (r, t) coordinates."""
    return (polar_pt[0]*r_scale), (polar_pt[1]+rot) % (2*np.pi)


@contextmanager
def suppress_stdout():
    """Context manager to suppress printing to stdout by piping into devnull; by Dave Smith:
    https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout  # Reset stdout when finished!


def extract_peak_coords(origin: tuple[float, float],
                        v_pcf: np.ndarray,
                        thresh: float = 0,
                        min_dist: int = 4) -> list[tuple[float, float]]:
    """Extracts peak coordinates from a vPCF via max filtering.
    Args:
        origin: The origin of the vPCF; peak coordinates will be transformed to be centered on the origin.
        v_pcf: The vPCF to extract peaks from.
        thresh: Minimum pixel value allowed to be considered a peak, passed to SingleOrigin.detect_peaks. Defualt is 0.
        min_dist: Minimum distance (px) allowed between peaks, passed to SingleOrigin.detect_peaks. Default is 4.

    Returns:
        A list of (x, y) coordinates of peak locations, centered on the origin.
    """
    peaks = np.argwhere(detect_peaks(v_pcf, min_dist=min_dist, thresh=thresh))
    peak_origin_coordinates = [((peak[0] - origin[0]) * 0.01, (peak[1] - origin[1]) * 0.01) for peak in peaks]
    return peak_origin_coordinates


def merge_close(coordinates: list[tuple[float, float]],
                tol: float,
                show_graph: bool = False)\
        -> list[tuple[float, float]]:
    """Merge closely-neighboring points into a single point, with coordinates at the center of mass of the neighborhood.
    Args:
        coordinates: A collection of cartesian point coordinates.
        tol: The maximum Euclidean distance to count as a neighbor.
        show_graph: Whether to plot the near-neighbor graph; useful for debugging. Default is False.
    Returns:
        The list of coordinates after merging.
    """
    # Algorithm: 1) Get the full distance matrix (fast for a small number of points)
    #            2) For each point, find all of its neighbors (within tol distance)
    #            3) Construct an undirected graph from the points (neighbors share an edge)
    #            4) Find all the connected components of the graph
    #            5) Add a new point at the center of mass of each connected component
    #            6) Remove the unmerged points
    dists = squareform(pdist(np.array(coordinates), metric="euclidean"))
    close_neighbor_edges = set()
    for i in range(len(coordinates)):
        neighbors = np.argwhere(dists[i, :] < tol).flatten()
        if neighbors.size == 1:  # Disregard points which only neighbor themselves
            continue
        # Set comprehension to ignore duplicate edges; frozenset is unordered and hashable
        new_edges = {frozenset([i, n]) for n in neighbors if n != i}
        close_neighbor_edges.update(new_edges)

    edgelist = list(map(tuple, close_neighbor_edges))  # Format as list of tuples for networkx to consume
    graph = from_edgelist(edgelist)
    if show_graph:
        draw_networkx(graph, node_size=10, arrowsize=20,
                      node_color="#ffffff", edge_color="#ffffff", font_color="#0088ff")
        messagebox.showinfo(title="networkx.draw(graph)",
                            message="Displaying point connectivity graph; press OK to continue.",
                            detail="Distances in connectivity graph do not correspond to Euclidean distances.")

    conn_comp = list(connected_components(graph))
    merged_coords, drop_indices = [], []
    for comp in conn_comp:
        merge_x, merge_y = None, None
        for index in comp:
            drop_indices.append(index)
            x, y = coordinates[index]
            if merge_x is None:  # Initial values
                merge_x = x
                merge_y = y
            else:
                merge_x += x
                merge_y += y
        merge_x /= len(comp)
        merge_y /= len(comp)
        merged_coords.append((merge_x, merge_y))

    final_coords = copy(coordinates)
    for i in sorted(drop_indices, reverse=True):  # Remove unmerged points from end first
        del final_coords[i]
    final_coords.extend(merged_coords)
    return final_coords


def polar_vpcf_peaks_from_cif(cif_path: Path | str,
                              axes: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
                              e: str,
                              e_ignore: str | list[str] | None = None,
                              xlims: tuple[float, float] = (-10, 10),
                              ylims: tuple[float, float] = (-10, 10),
                              px_size: float = 0.01,
                              uc_merge_tol: float = 1,
                              vpcf_tol: float = 0.5)\
        -> list[tuple[float, float]]:
    """Generate the ideal peak coordinates of a vPCF given a cif file and zone axis, in polar form.
    Args:
        cif_path: The path to the cif file (string or pathlib Path object).
        axes: A tuple containing the zone axis to project, as well as orthogonal basis vectors to complete
            the coordinate system.  The format is (zone axis, basis 2, basis 3), and each vector should be a
            tuple of three integers.  Example: ((0, 0, 1), (5, 0, 0), (0, 5, 0)).
        e: Symbol of the element of interest (e.g. "Al").
        e_ignore: Symbol of the element to ignore when generating the unit cell (e.g. "N"); can also be a
            list of symbols or None. Default is None.
        xlims: The x-limits for the vPCF. xlims[0] should be less than xlims[1], and the range must include 0.
            Default is (-10, 10).
        ylims: The y-limits for the vPCF. ylims[0] should be less than ylims[1], and the range must include 0.
            Default is (-10, 10).
        px_size: The pixel size of the vPCF, in the same units as the unit cell atom column coordinates.
            Default is 0.01.
        uc_merge_tol: The tolerance for merging nearby atom columns when projecting the unit cell (A).
            Default is 1.
        vpcf_tol: The tolerance for merging nearby peaks in the vPCF (A). Default is 0.5.
    Returns:
        The peaks of the vPCF calculated from the cif file along the given zone axis, in polar coordinates.
    """
    za, a1, a2 = axes
    uc = UnitCell(str(cif_path))  # Must cast path object to string for SingleOrigin to handle it
    # SingleOrigin expects ignore_elements to always be a list
    if type(e_ignore) is str:
        e_ignore = [e_ignore]
    elif e_ignore is None:
        e_ignore = []

    with suppress_stdout():
        uc.project_zone_axis(za, a1, a2,
                             ignore_elements=e_ignore,
                             reduce_proj_cell=False)
        uc.combine_prox_cols(toler=uc_merge_tol)
    vpcf, origin = get_vpcf(xlim=xlims, ylim=ylims, d=px_size,
                            coords1=uc.at_cols.loc[uc.at_cols["elem"] == e, ["x", "y"]].to_numpy())
    # We need to transpose the vpcf to get the same coordinate system as the experimental peaks
    peak_origin_coordinates = extract_peak_coords(origin, vpcf.T)  # Default thresh & min_dist is fine for ref peaks
    merged_coordinates = merge_close(peak_origin_coordinates, vpcf_tol)
    return [xy_to_rt(xy) for xy in merged_coordinates]


def test_stack(analysis_stack: np.ndarray) -> None:
    global fig, axs, ax, i
    fig, axs = plt.subplots(4, 4, figsize=(4, 4),
                            subplot_kw={"xticks": [], "yticks": []},
                            gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    for ax in axs.flat:
        i = np.random.randint(len(analysis_stack))
        ax.imshow(analysis_stack[i], cmap="inferno")


def select_files(single: bool = False, ftypes: list[tuple[str, str]] = None):
    """Helper function for selecting file(s) with a Tkinter dialog which forces the window to be focused on creation.
    Unfortunately the window may still draw behind certain applications (no idea how to fix, happens sporadically),
    but at least this way we get a blinking icon in the taskbar.

    Args:
        single: If True, calls askopenfilename. If False, calls askopenfilenames.
        ftypes: List of allowed filetypes, in the normal format used by tkinter filedialogs.

    Returns:
        The filename or list of filenames.
    """
    root = tkinter.Tk()
    root.iconify()
    root.focus_force()

    if single is True:
        try:
            return askopenfilename(parent=root, filetypes=ftypes)
        finally:
            root.destroy()
    else:
        try:
            return askopenfilenames(parent=root, filetypes=ftypes)
        finally:
            root.destroy()


def update_dmax(good_fit_tol: float,
                mu: float,
                sig: float,
                prefactor: float = 20) -> float:
    """Update the maximum tolerable distance based on statistics about the distance distribution.
    Args:
        good_fit_tol: User-specified goodness-of-fit parameter.
        mu: The mean of the distance distribution.
        sig: The standard deviation of the distance distribution.
        prefactor: Multiplier for good_fit_tol used when the fit is bad. Default is 20 (see Zhang 1993).
    Returns:
        The new d_max value.
    """
    if mu < good_fit_tol:  # Good registration
        return mu + 3*sig
    elif mu < 3*good_fit_tol:  # Decent registration
        return mu + 2*sig
    elif mu < 6*good_fit_tol:  # Okay-ish registration
        return mu + sig
    else:  # mu >= 6*good_fit_tol
        # Bad fit; we need to resort to back-up d_max computation
        # Zhang 1993 does this by choosing xi to be in the valley to the right of the modal distance (for some
        # histogram binning). We have too few points to reasonably bin, so instead we fall back to an initial
        # guess of prefactor*good_fit_tol.
        return prefactor*good_fit_tol


def loss(params: tuple[float, float],
         pts: list[tuple[float, float]],
         tree: spatial.KDTree,
         good_fit_tol: float,
         is_coarse: bool = False) -> float:
    """Loss function which can either dynamically update d_max (for fine fitting) or not (for coarse global fitting).
    Args:
        params: Tuple of the form (r_scale, rot), which are the fitting parameters to be optimized.
        pts: The list of points currently being fit.
        tree: A k-D tree representing the points currently being fit to.
        good_fit_tol: User-specified goodness-of-fit parameter, used to calculate d_max.
        is_coarse: Set to True to disable dynamic updates of d_max (for coarse search).

    Returns:
        The loss for the current fitting parameters.
    """
    try:
        exp_transformed = [similarity_transform(pt, params[0], params[1]) for pt in pts]
        # We can't just use distance_upper_bound param of query, since it assigns d=inf to unmatched points
        ds, _ = tree.query(exp_transformed)
        if not is_coarse:  # Dynamically update d_max during iteration for fine fitting
            mu, sig = np.mean(ds), np.std(ds)
            d_max = update_dmax(good_fit_tol, mu, sig)
            ds = [d for d in ds if d <= d_max]

        if len(ds) == 0:
            return np.inf
        else:
            return sum(ds) / len(ds)

    except ValueError:
        # Some optimize methods (e.g. L-BFGS-B) seem to occasionally try (nan, nan) as fit parameters
        # I don't know why... but at least this catches that behavior
        print(f"NaN encountered in fit parameters: {params}")


def coarse_search(bounds: tuple[tuple[float, float], tuple[float, float]],
                  n_r: int, n_t: int,
                  pts: list[tuple[float, float]],
                  tree: spatial.KDTree,
                  good_fit_tol: float,
                  return_all: bool = False)\
        -> tuple[float, float, float] | tuple[float, float, float, list[tuple[float, float, float]]]:
    """Grid-based brute-force search over (scale, rotation) parameters, meant for coarse global optimization.
    Args:
        bounds: Tuples containing the bounds of the search. Scale bounds first, rotation bounds second.
        n_r: The number of search points to generate on the scale axis between the given bounds (inclusive).
        n_t: The number of search points to generate on the rotation axis between the given bounds (inclusive).
        pts: The set of points used to query the tree for the overall loss.
        tree: The tree to be queried for the overall loss.
        good_fit_tol: User-defined goodness-of-fit parameter.
        return_all: If True, returns a list of all checked (scale, rotation, loss) values, useful for mapping the
            optimization space. Default is False.

    Returns:
        The scale and rotaiton parameters associated with the minimum loss value, as well as that loss value. If
            return_all is True, additionally returns a list of all checked (scale, rotation, loss) points.
    """
    r_scale_span = np.linspace(bounds[0][0], bounds[0][1], n_r, endpoint=True)
    rot_span = np.linspace(bounds[1][0], bounds[1][1], n_t, endpoint=True)

    minimum, min_r, min_t = np.inf, 1, 0
    collector = []
    for r in r_scale_span:
        for t in rot_span:
            _l = loss((r, t), pts, tree, good_fit_tol, is_coarse=True)
            if return_all:
                collector.append((r, t, _l))
            if _l < minimum:
                minimum = _l
                min_r, min_t = r, t
    if not return_all:
        return min_r, min_t, minimum
    else:
        return min_r, min_t, minimum, collector


def simplex_refine(pts: list[tuple[float, float]], tree: spatial.KDTree, good_fit_tol: float,
                   r_init: float, t_init: float, lookaround: tuple[float, float],
                   alpha: float = 1, gamma: float = 2, rho: float = 0.5, sigma: float = 0.5,
                   break_thresh: float = 1e-4, break_iter: int = 30, max_iter: int = 10000,
                   r_bounds: tuple[float, float] | None = None, t_bounds: tuple[float, float] | None = None,
                   return_track: bool = False)\
        -> tuple[float, float] | tuple[float, float, list[tuple[float, float, float]]]:
    """Refine the scale and roation values using the Nelder-Mead simplex method.
    Args:
        pts: The list of points being fitted (polar coordinates).
        tree: The KDTree against which the points are being fitted.
        good_fit_tol: User-defined goodness-of-fit parameter.
        r_init: The initial scaling factor; from a coarse search, or else user specified.
        t_init: The initial rotation offset; from a coarse searach, or else user specified.
        lookaround: The distance to look along each axis (in the + direction) when constructing the initial simplex.
        alpha: Refletion coefficient (must be > 0). Larger values send the reflected point further from the centroid.
            Default is 1.
        gamma: Expansion coefficient (must be > 1). Larger values send the xepanded point further from the centroid.
            Default is 2.
        rho: Contraction coefficient (must be between 0 and 0.5). Larger values leave the contracted point further from
            the centroid. Default is 0.5.
        sigma: Shrink coefficient (must be between 0 and 1). Larger values shrink the simplex _less_. Default is 0.5.
        break_thresh: The threshold for per-iteration improvement; the search terminates if the improvement is less
            than this value for `break_iter` iterations. Default is 1e-4.
        break_iter: The number of iterations after which - if improvements are below the threshold value - the search
            will terminate. Default is 30.
        max_iter: The absolute maximum number of iterations allowed. If the loop breaks this way, a warning message
            will be emitted before returning as usual. Default is 10000.
        r_bounds: Optional bounds on allowed scaling values. Default is None.
        t_bounds: Optional bounds on allowed rotation values. Default is None.
        return_track: If True, return a list of all best points explored during iteration, and the loss values
            at each such point. The final point in this list will always be the optimized solution.

    Returns:
        The refined (scale, rotation) values, as well as the loss at that point.  If return_track is True,
            also returns a list of (scale, rotation, loss) values that follows the centroid of the simplex.
    """
    # This code is not the most pythonic, and it only works for 2 dimensions, but otherwise pretty closely follows
    #  the reference implementation on Wikipedia.

    # Parameter checks
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, not {alpha}")
    if gamma <= 1:
        raise ValueError(f"gamma must be greater than 1, not {gamma}")
    if rho <= 0 or rho > 0.5:
        raise ValueError(f"rho must be in the range (0, 0.5], not {rho}")
    if sigma <= 0 or sigma > 1:
        raise ValueError(f"sigma must be in the range (0, 1], not {sigma}")
    if break_thresh <= 0:
        raise ValueError(f"break_thresh must be greater than 0, not {break_thresh}")
    if break_iter <= 0:
        raise ValueError(f"break_iter must be greater than 0, not {break_iter}")
    if type(r_bounds) is not None and (r_init < r_bounds[0] or r_init > r_bounds[1]):
        raise ValueError(f"r_bounds {r_bounds} must be inclusive of initial guess {r_init}")
    if type(t_bounds) is not None and (t_init < t_bounds[0] or t_init > t_bounds[1]):
        raise ValueError(f"t_bounds {t_bounds} must be inclusive of initial guess {t_init}")

    # Create initial simplex
    x0 = (r_init, t_init, loss((r_init, t_init), pts, tree, good_fit_tol))
    x1 = (r_init + lookaround[0], t_init, loss((r_init + lookaround[0], t_init), pts, tree, good_fit_tol))
    x2 = (r_init, t_init + lookaround[1], loss((r_init, t_init + lookaround[1]), pts, tree, good_fit_tol))
    simplex = [x0, x1, x2]
    track = []  # Will only be used if return_track is True

    # Iteration
    total_iter = -1
    iter_without_improvement = 0  # Used in termination check
    while True:
        # Sort simplex vertices by loss
        simplex = sorted(simplex, key=lambda x: x[-1])

        if return_track:
            track.append(simplex[0])

        # Check for convergence / termination
        total_iter += 1
        if total_iter >= max_iter:
            warn("Iteration ended due to reaching maximum iteration count; result may not be converged!")
            break
        if simplex[0][-1] < (simplex[1][-1] - break_thresh):
            iter_without_improvement = 0
        else:  # Best two guesses are closer together than break_thresh
            iter_without_improvement += 1
        if iter_without_improvement >= break_iter:
            break

        xx = ((x0[0]+x1[0])/2, (x0[1]+x1[1])/2)  # Midpoint of the line connecting best two guesses

        # Reflection
        xr = (xx[0] + alpha*(xx[0] - simplex[-1][0]),  # reflected point
              xx[1] + alpha*(xx[1] - simplex[-1][1]))
        xrl = loss((xr[0], xr[1]), pts, tree, good_fit_tol)
        if simplex[0][-1] < xrl < simplex[-2][-1]:
            _ = simplex.pop()
            simplex.append((xr[0], xr[1], xrl))
            continue

        # Expansion
        if xrl < simplex[0][-1]:
            xe = (xx[0] + gamma*(xr[0] - xx[0]),  # expanded point
                  xx[1] + gamma*(xr[1] - xx[1]))
            xel = loss((xe[0], xe[1]), pts, tree, good_fit_tol)
            if xel < xrl:
                _ = simplex.pop()
                simplex.append((xe[0], xe[1], xel))
                continue
            else:
                _ = simplex.pop()
                simplex.append((xr[0], xr[1], xrl))
                continue

        # Contraction
        # If we reach this point, we know xrl >= simplex[-2][-1] (second worst point)
        if xrl < simplex[-1][-1]:
            xco = ((xx[0] + rho*(xr[0] - xx[0])),  # outer contracted point
                  (xx[1] + rho*(xr[1] - xr[1])))
            xcol = loss((xco[0], xco[1]), pts, tree, good_fit_tol)
            if xcol < xrl:
                _ = simplex.pop()
                simplex.append((xco[0], xco[1], xcol))
                continue
        elif xrl >= simplex[-1][-1]:
            xci = ((xx[0] + rho*(simplex[-1][0] - xx[0])),  # inner contracted point
                   (xx[1] + rho*(simplex[-1][1] - xx[1])))
            xcil = loss((xci[0], xci[1]), pts, tree, good_fit_tol)
            if xcil < simplex[-1][-1]:
                _ = simplex.pop()
                simplex.append((xci[0], xci[1], xcil))
                continue

        # Shrink
        new_pts = []
        for vert in simplex[1:]:
            xi_new = ((simplex[0][0] + sigma*(vert[0] - simplex[0][0])),
                      (simplex[0][1] + sigma*(vert[1] - simplex[0][1])))
            xi_new_l = loss((xi_new[0], xi_new[1]), pts, tree, good_fit_tol)
            new_pts.append((xi_new[1], xi_new[1], xi_new_l))
        _, _ = simplex.pop(), simplex.pop()  # Drop last two points...
        simplex += new_pts  # ...and append the shrunk points

    r_final, t_final = simplex[0][0], simplex[0][1]
    if return_track:
        # noinspection PyUnboundLocalVariable
        # If we get here, track will always be assigned
        return r_final, t_final, track
    else:
        return r_final, t_final


# %% Open .cif files for each relevant phase
element_of_interest: str = "Hf"  # Currently only supporting single-site vPCFs
ignore_elements: list[str] = ["O"]  # The listed elements will be skipped when generating vPCF coordinates

file_list = [Path(file) for file in
             select_files(ftypes=[("Crystallographic Information File", ".cif")])]
print([f"{i}: {f.stem}" for i, f in enumerate(file_list)])

# %% SingleOrigin setup
# The following code contains some hard-coded values, and may need to be adjusted for other structures!
fitting_file: Path = file_list[2]  # Change to select the .cif file to use for initial fitting

uc = UnitCell(str(fitting_file), origin_shift=(0.1, 0.1, 0))
uc.project_zone_axis((0, 1, 1),  # Zone axis direction
                     (0, 2, 0),
                     (2, 0, 0),
                     ignore_elements=ignore_elements,
                     reduce_proj_cell=False)
uc.combine_prox_cols(toler=1)
# so.load_img supports .dm4, .dm3, .emd, .ser, .tif, .png, .jpg
img_path = select_files(single=True, ftypes=[("Digital Micrograph 3", ".dm3"), ("Digital Micrograph 4", ".dm4"),
                                             ("Electron Microscopy Dataset", ".emd"), ("Serial", ".ser"),
                                             ("Tagged Image File Format", ".tif"), ("JPEG", ".jpg"),
                                             ("Portable Network Graphics", ".png")])
if not img_path == "":  # File selection was canceled; continuing will kill the python process, so don't
    image, _ = load_image(path=img_path, display_image=False, images_from_stack="all")
else:
    raise RuntimeError("No file selected!")
hrimage = HRImage(image)
lattice = hrimage.add_lattice("lattice", uc)
lattice.get_roi_mask_polygon()

# %% Fit lattice to image, and refine
plot_fit_lattice: bool = False  # Whether to plot the fit lattice, to make sure column positions look right

lattice.fft_get_basis_vect(a1_order=4, a2_order=4, sigma=3)
lattice.define_reference_lattice(plot_ref_lattice=False)
lattice.fit_atom_columns(buffer=10, bkgd_thresh_factor=0.0, parallelize=False,
                         use_circ_gauss=False, watershed_line=True, peak_grouping_filter=None)

if plot_fit_lattice:
    hrimage.plot_atom_column_positions(fit_or_ref='fit', outlier_disp_cutoff=100, scalebar_len_nm=None)

# %% Extract sub-images and local column coordinates for each
window_size: int = 256  # Set kernel size (px)
test_subimages: bool = False  # Whether to show a few sub-images as a sanity check

raw = image.reshape(1, image.shape[0], image.shape[1])
lattice.at_cols["class"] = 0  # Class number, required by atomai
coords = lattice.at_cols[["x_fit", "y_fit", "class"]].to_numpy()
# noinspection PyTupleAssignmentBalance
# Type signature used for return value of extract_subimages should be tuple[list, list, list[ndarray]] or similar
imstack, coms, _ = utils.extract_subimages(np.expand_dims(raw, axis=3),
                                           coordinates={0: np.array(coords)},
                                           window_size=window_size,
                                           coord_class=0)
imstack = imstack[..., 0]  # Disregard channel dimension

if test_subimages:
    test_stack(imstack)

# %% Select what will be used for k-means cluster analysis
test_vpcfs: bool = False  # Whether to show some vPCFas to verify things are working as expected

coords_around_point = []
for frame in range(imstack.shape[0]):
    given_point = coms[frame]
    # Extracting coordinates around the given point
    distances = np.sqrt((lattice.at_cols["x_fit"] - given_point[0]) ** 2 +
                        (lattice.at_cols["y_fit"]-given_point[1]) ** 2)
    indices = distances[distances <= window_size].index
    coords_around_point.append(lattice.at_cols.loc[indices])

lattice_copy = copy(lattice)
vpcf = []
for i in tqdm(range(len(coords_around_point)),
              desc="Getting subimage vPCFs", unit="vPCFs"):
    lattice_copy.at_cols = coords_around_point[i]
    # TODO: Resulting vPCFs should be square; set this up to do that automatically
    with suppress_stdout():  # Doing so many clutters the console and all the printing slows the code way down
        lattice_copy.get_vpcfs(xlim=[-1, 1],
                               ylim=[-0.679, 0.679],
                               d=0.0555)
    vpcf_name = f"{element_of_interest}-{element_of_interest}"
    vpcf.append(lattice_copy.vpcfs[vpcf_name])
vpcfs = np.array(vpcf)

if not vpcfs.shape[1] == vpcfs.shape[2]:
    print(f"vPCFs are not square!  Shape: {vpcfs.shape}")
analysis_stack = vpcfs[:, :-1, :-1]  # TODO: Re-size vPCFs to be window_size?

if test_vpcfs:
    test_stack(analysis_stack)

# %% k-means clustering
# TODO: Maybe try DBSCAN?
num_clusters: int = 4  # Hyperparameter for k-means: number of clusters in image
show_kmeans_central_members: bool = False  # Whether to show representative (central) vPCFs for each k-means cluster
show_kmeans_map: bool = False  # Whether to show the spatial map of k-means clusters

z = analysis_stack.reshape(-1, window_size**2)
try:
    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=0, n_init=10, verbose=0).fit(z)
except Warning:
    # If we load too much data into the k-means fitting, it will throw a bunch of resource_tracker warnings and then
    # kill the entire python process.  We'll throw an error instead to preserve the process and allow a re-try with
    # a new window size, without re-running the entire script.
    # TODO: I need to actually test that catching the warning this way keeps the process alive
    raise RuntimeError("Too much data in memory!")

if show_kmeans_central_members:
    rows = int(np.ceil(float(num_clusters) / 4))
    cols = int(np.ceil(float(num_clusters) / rows))
    gs = matplotlib.gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(4*cols, 4*(1+rows//1.5)))
    for i in range(num_clusters):
        ax = fig.add_subplot(gs[i])
        # noinspection PyUnresolvedReferences
        # Seems like if attributes are assigned after fitting, pycharm can't see them
        ax.imshow(kmeans.cluster_centers_[i, :].reshape(window_size, window_size), cmap='inferno')
        ax.set_title('Cluster Center - ' + str(i + 1))
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
    plt.show()

if show_kmeans_map:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    ax.imshow(raw[0], cmap="gray")
    # noinspection PyUnresolvedReferences
    scatter = ax.scatter(coms[:, 0], coms[:, 1], c=kmeans.labels_, cmap="jet", alpha=1)
    ax.set_title('kMeans labels')
    ax.set_axis_off()
    plt.colorbar(scatter)
    plt.show()

# %% Generate total vPCFs for each cluster
# This is technically a bit different from just the central members, and should give better accuracy
show_total_vpcfs: bool = True  # Whether to show the total vPCFs
show_exp_peaks: bool = True  # If true, plot the located peaks over the vPCF images (show_total_vpcfs must be True)
exp_vpcf_tol: float = 0  # Tolerance for merging nearby _experimental_ vPCF peaks; scale may differ from ref vPCFs
exp_peak_thresh: float = 100  # Minimum pixel value allowed to count as a vPCF peak; increase to filter out noise
exp_peak_min_dist: int = 10  # Minimum distance allowed between experimental vPCF peaks

# TODO: Something seems to have broken here, the identified experimental vPCF peaks are not quite in the right place...

lattice_copy_2 = copy(lattice)
hrimage_2 = copy(hrimage)
# noinspection PyUnresolvedReferences
labels = np.unique(kmeans.labels_)
vpcf_clusters = []
vpcf_origins = []
for i in range(labels.shape[0]):
    # noinspection PyUnresolvedReferences
    coor_cluster = coms[np.where(kmeans.labels_ == i)[0]]
    total_cluster = lattice.at_cols[(lattice.at_cols['x_fit'].isin(coor_cluster[:, 0]))
                                    & (lattice.at_cols['y_fit'].isin(coor_cluster[:, 1]))]
    lattice_copy_2.at_cols = total_cluster
    with suppress_stdout():
        lattice_copy_2.get_vpcfs(xlim=[-1, 1],
                                 ylim=[-0.679, 0.679],
                                 d=0.0555)
    vpcf_name = f"{element_of_interest}-{element_of_interest}"
    vpcf_clusters.append(lattice_copy_2.vpcfs[vpcf_name])
    vpcf_origins.append(lattice_copy_2.vpcfs["metadata"]["origin"])

exp_peaks = []
for vc, o in zip(vpcf_clusters, vpcf_origins):
    peaks = np.array([xy_to_rt(pt) for pt in
                      merge_close(extract_peak_coords(o, vc.T,
                                                      thresh=exp_peak_thresh,
                                                      min_dist=exp_peak_min_dist),
                                  exp_vpcf_tol)])
    exp_peaks.append(peaks)

if show_total_vpcfs:
    fig, axs = plt.subplots(nrows=1, ncols=num_clusters, figsize=(4*num_clusters, 4))
    if not show_exp_peaks:
        for i, clust in enumerate(vpcf_clusters):
            axs[i].imshow(clust, cmap="inferno")
            axs[i].axes.get_xaxis().set_ticks([])
            axs[i].axes.get_yaxis().set_ticks([])
            axs[i].set_title('Cluster Center - ' + str(i + 1))
    else:
        for i, (clust, (peaks, ori)) in enumerate(zip(vpcf_clusters, zip(exp_peaks, vpcf_origins))):
            axs[i].imshow(clust, cmap="inferno")
            axs[i].axes.get_xaxis().set_ticks([])
            axs[i].axes.get_yaxis().set_ticks([])
            axs[i].scatter(ori[0], ori[1], marker="+", c="#ffffff", s=30)
            # Transform exp_peaks back to image coordinates
            ps = np.array([rt_to_xy(p) for p in peaks])
            p_x, p_y = ps[:, 0], ps[:, 1]
            pxs = [p/0.01+ori[0] for p in p_x]
            pys = [p/0.01+ori[1] for p in p_y]
            axs[i].scatter(pxs, pys, marker="+", c="#fe6100", s=20)
            axs[i].set_title('Experimental Peaks - ' + str(i + 1))


# %% Generate vPCF coordinates for the perfect structures
uc_tol: float = 1  # Tolerance for merging nearby atom columns in the projected unit cell (A)
vpcf_tol: float = 0.5  # Tolerance for merging nearby vPCF peaks (A)

# Tuples are (zone axis vector, basis vector 2, basis vector 3)
# TODO: This currently uses the same set of zone axes for all phases, which may not be the correct behavior
zones = [((0, 0, 1), (5, 0, 0), (0, 5, 0)),
         ((0, 1, 0), (5, 0, 0), (0, 0, -5)),
         ((1, 0, 0), (0, 0, -5), (0, 5, 0)),
         ((1, 1, 0), (5, -5, 0), (0, 0, -5)),
         ((1, 0, 1), (5, 0, -5), (0, -5, 0)),
         ((0, 1, 1), (0, -5, 5), (5, 0, 0)),
         ((1, 1, 1), (5, 0, -5), (-5, 10, -5))]

_temp_mapping = [(fname, axes) for fname in file_list for axes in zones]
struct_mapping = {}  # Dict from name (phase & orientation) to corresponding vPCF coordinate lists
for mapping in tqdm(_temp_mapping, desc="Generating refernce vPCFs", unit="vPCF"):
    map_name = str((mapping[0].stem, mapping[1][0]))
    # TODO: Currently using default values for xlims, ylim, px_size -- might want to adjust?
    struct_mapping[map_name] = polar_vpcf_peaks_from_cif(mapping[0], mapping[1],
                                                         e=element_of_interest, e_ignore=ignore_elements,
                                                         uc_merge_tol=uc_tol, vpcf_tol=vpcf_tol)
del _temp_mapping

# %% Build k-D trees for each set of reference points
reference_forest = {}
for k, ref_peaks in struct_mapping.items():
    reference_forest[k] = spatial.KDTree(ref_peaks)

# %% Optionally show all reference vPCFs for visual inspection of reasonableness
test_refs: bool = False  # Set this to enable displaying coordinate plots (useful for debugging)
if test_refs:
    for struct_map in struct_mapping.items():
        peaks = struct_map[1]
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        ax.set_rlim(rmin=0, rmax=np.ceil(np.max(np.array(list(peaks))[:, 0])))
        ax.set_thetagrids((0, 90, 180, 270), ("0", u"\u03c0/2", u"\u03c0", "3"+u"\u03c0/2"))
        ax.scatter(np.array(list(peaks))[:, 1], np.array(list(peaks))[:, 0])
        fig.suptitle(struct_map[0])
        fig.show()


# %% Fit modeled vPCFs to total vPCFs
r_scale_bounds: tuple[float, float] = (0.5, 7)  # Lower and upper bounds on scaling
rot_bounds: tuple[float, float] = (0, 2*np.pi)  # Lower and upper bounds on rotation
good_fit_tol: float = 5e-2  # User-specified goodness-of-fit parameter; approx. expected mean dist. when fit is good
dmax_prefactor: int = 100  # Prefactor for d_max when fit is poor (and for initial/coarse search); should be large-ish
# Zhang 1993 uses 20 as their prefactor, but for this application probably higher (initial mismatch may be large)

##################
# IMPORTANT NOTE # Currently set up to match reference vPCFs with themselves
##################

print("Fitting (global)...")
scale_init, rot_init, _, loss_map = coarse_search((r_scale_bounds, rot_bounds), 500, 500,
                                                  struct_mapping["('Monoclinic', (1, 0, 1))"],
                                                  reference_forest["('Monoclinic', (1, 0, 1))"],
                                                  good_fit_tol, return_all=True)

print("Refining (local)...")
# r_fit, t_fit, track = simplex_refine(struct_mapping["('Monoclinic', (1, 0, 1))"],
#                                      reference_forest["('Monoclinic', (1, 0, 1))"],
#                                      good_fit_tol, scale_init, rot_init, (-0.1, -0.1),
#                                      alpha=0.1, gamma=1.1, rho=0.1, sigma=0.1,
#                                      r_bounds=r_scale_bounds, t_bounds=rot_bounds, return_track=True)
optimized = optimize.minimize(loss,
                              x0=np.array([scale_init, rot_init]),
                              args=(struct_mapping["('Monoclinic', (1, 0, 1))"],
                                    reference_forest["('Monoclinic', (1, 0, 1))"],
                                    good_fit_tol, False),
                              method="Nelder-Mead", bounds=(r_scale_bounds, rot_bounds))
print("Done!")

# %% Plotting function for debugging: view the loss landscape (based on coarse search)
# loss_map = np.array(loss_map)
# scales, rots = sorted(list(set(loss_map[:, 0]))), sorted(list(set(loss_map[:, 1])))
# losses = loss_map[:, 2].reshape(len(scales), len(rots)).T
# plt.contourf(scales, rots, losses, cmap="inferno")
# plt.colorbar()
# plt.xlabel("Radial Scale")
# plt.ylabel("Rotation")
# plt.show()
#%%
optimization_results = []
for i, pts in enumerate(struct_mapping.values()):
    optimized = {}
    for key, tree in tqdm(reference_forest.items(),
                          desc=f"Fitting vPCF {i+1}/{len(struct_mapping)}"):
        r_init, t_init, _ = coarse_search((r_scale_bounds, rot_bounds), 100, 100,
                                          pts, tree, good_fit_tol)

        optimized[key] = optimize.minimize(loss,
                                           x0=np.array([r_init, t_init]),
                                           args=(pts, tree, good_fit_tol, False),
                                           method="Nelder-Mead",  # L-BFGS-B can return NaNs!
                                           bounds=(r_scale_bounds, rot_bounds))

    optimization_results.append(optimized)


# %% Find the best fitting result for each vPCF and store for plotting
best_fitted = {}
for i, res in enumerate(optimization_results):
    best_fit, best_fitting = np.inf, ""
    for ref_fit in res.items():
        if ref_fit[1]["fun"] < best_fit:
            best_fit = ref_fit[1]["fun"]
            best_fitting = ref_fit[0]
        elif np.isclose(ref_fit[1]["fun"], best_fit):
            warn("Multiple best fits!")  # optimization_results still has all values
    print(f"Best fit for experimental vPCF {i} is {best_fitting} with loss {best_fit:.8f}.")
    best_fitted[i] = (best_fitting, res[best_fitting])

# %% Plot best-fitting vPCF peaks with experimental vPCFs
# noinspection PyUnresolvedReferences
# I guess pycharm doesn't know dicts have keys now???
for i, fitted in enumerate(best_fitted.keys()):
    r_scale = best_fitted[fitted][1]["x"][0]
    rot = best_fitted[fitted][1]["x"][1]
    ref_xy = [rt_to_xy(pt) for pt in struct_mapping[best_fitted[fitted][0]]]
    exp_xy = [rt_to_xy(similarity_transform(pt, r_scale, rot)) for pt in list(struct_mapping.values())[i]]
    fig, ax = plt.subplots()

    ax.scatter(np.array(ref_xy)[:, 0], np.array(ref_xy)[:, 1], marker="x",
               label=f"Reference: {best_fitted[fitted][0]}", c="#785ef0")
    ax.scatter(np.array(exp_xy)[:, 0], np.array(exp_xy)[:, 1], marker="+",
               label="Experimental", c="#fe6100")
    fig.legend()
    fig.show()

# %% Plot loss grid
grid = np.array([[res["fun"] for res in ress.values()] for ress in optimization_results])
symmetry = np.ones(grid.shape)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        fst, snd = grid[i][j], grid[j][i]
        if not np.isclose(fst, snd, atol=0.1):
            symmetry[i][j] = 0
fig, axs = plt.subplots(1, 2)
axs[0].imshow(grid, cmap="cividis", origin="lower")
axs[1].imshow(symmetry, cmap="cividis", origin="lower")
# plt.colorbar()
plt.show()

# %% Plot loss distributions
cmap_min = min([opt["fun"] for res in optimization_results for opt in res.values()])
cmap_max = np.max([opt["fun"] for res in optimization_results for opt in res.values()
                   if not np.isinf(opt["fun"])])
median = np.median([opt["fun"] for res in optimization_results for opt in res.values()
                    if not np.isinf(opt["fun"])])
fig, axs = plt.subplots(1, len(struct_mapping), sharey="all")
cmap = plt.get_cmap("RdYlGn_r")
axs[0].set_ylabel("Loss (a.u.)")
axs[0].set_ylim((0, median*1.1))
for i, res in enumerate(optimization_results):
    axs[i].set_xticks([])
    axs[i].spines[["bottom", "top", "right"]].set_visible(False)
    for j, (phase_orientation, opt) in enumerate(sorted(res.items(), key=lambda x: x[1]["fun"])):
        axs[i].bar(x=j, height=opt["fun"],
                   color=cmap((opt["fun"] - cmap_min) / (cmap_max - cmap_min)))
        axs[i].set_title(f"{i}")
fig.show()

# %%
