# %%
import copy
from tkinter.filedialog import askopenfilenames, askopenfilename
from tkinter import messagebox
from pathlib import Path
from typing import Literal
import numpy as np
import SingleOrigin as so
import matplotlib
import matplotlib.pyplot as plt
import scipy.spatial.distance as scidist
import warnings
from scipy import optimize
from sklearn import cluster
import atomai
import networkx as nx

matplotlib.use("TkAgg")  # Needed to get interactive plots in PyCharm
plt.style.use('dark_background')
plt.rcParams.update({"figure.max_open_warning": 0})  # Suppress warnings when testing vPCF output


def xy_to_rt(xy: tuple) -> tuple:
    """Convert a single point from Cartesian to polar coordinates."""
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
    """Convert a single point from polar to Cartesian coordinates."""
    x = rt[0]*np.cos(rt[1])
    y = rt[0]*np.sin(rt[1])
    return x, y


def polar_vpcf_peaks_from_cif(cif_path: Path | str,
                              axes: tuple[tuple[int, int, int]],
                              e: str,
                              uc_merge_tol: float = 1,
                              vpcf_tol: float = 0.5)\
        -> list[tuple[float, float]]:
    """Generate the ideal peak coordinates of a vPCF given a cif file and zone axis, in polar form.
    Args:
        cif_path: The path to the cif file (string or pathlib Path object).
        axes: A tuple containing the zone axis to project, as well as orthogonal basis vectors to complete
            the coordinate system.  The format is (zone axis, basis 2, basis 3), and each vector should be a
            tuple of three integers.  Example: ((0, 0, 1), (5, 0, 0), (0, 5, 0)).
        e: Symbol of the element of interest (e.g. "Al")
        uc_merge_tol: The tolerance for merging nearby atom columns when projecting the unit cell (A).
        vpcf_tol: The tolerance for merging nearby peaks in the vPCF (A).
    Returns:
        The peaks of the vPCF calculated from the cif file along the given zone axis, in polar coordinates.
    """
    # noinspection PyTupleAssignmentBalance
    za, a1, a2 = axes
    uc = so.UnitCell(str(cif_path))  # Must cast path object to string for SingleOrigin to handle it
    uc.project_zone_axis(za, a1, a2,
                         ignore_elements=ignore_elements,
                         reduce_proj_cell=False)
    uc.combine_prox_cols(toler=uc_merge_tol)

    v_pcf, origin = so.v_pcf(xlim=(-6, 6), ylim=(-6, 6), d=0.01,
                             coords1=uc.at_cols.loc[uc.at_cols["elem"] == e, ["x", "y"]].to_numpy())
    peak_origin_coordinates = extract_peak_coords(origin, v_pcf)
    merged_coordinates = merge_close(peak_origin_coordinates, vpcf_tol)
    return [xy_to_rt(xy) for xy in merged_coordinates]


def extract_peak_coords(origin: tuple[float, float], v_pcf: np.ndarray) -> list[tuple[float, float]]:
    peaks = np.argwhere(so.detect_peaks(v_pcf, min_dist=4))
    peak_origin_coordinates = [((peak[0] - origin[0]) * 0.01, (peak[1] - origin[1]) * 0.01) for peak in peaks]
    return peak_origin_coordinates


def loss(a, b):
    # Lists must be sorted so that permutations of point order do not affect the loss function
    _a_sorted = np.array(sorted(a, key=lambda x: [x[0], x[1]]))  # Sort radially, break ties by theta
    _b_sorted = np.array(sorted(b, key=lambda x: [x[0], x[1]]))
    _dist_matrix = scidist.cdist(_a_sorted, _b_sorted, metric="sqeuclidean")  # Squared-euclidean distance
    return np.trace(_dist_matrix)


def optimize_wrapper(params: np.ndarray,
                     ref_peaks: list[tuple[float, float]],
                     exp_peaks: list[tuple[float, float]]) -> float:
    """The function to be passed to scipy.optimize.minimize - performs a similarity transform then returns the loss.
    A similarity
    Args:
        params: An ndarray of shape (2,) containing the initial guesses for the scale and rotation factors
            (typically [1, 0]).  The (radial) scale factor params[0] must be a positive float,
            and the rotation factor params[1] (in radians) is automatically scaled to be in the range [0, 2pi).
        ref_peaks: The peaks in the reference vPCF, which will be transformed according to scale and rotation to
            fit the experimental peaks as well as possible.
        exp_peaks: The experimental peaks.
    Returns:
        The loss from summing the squared distance between each experimental point and the nearest reference point.
        """
    _rscale, _rot = params[0], params[1]
    ref_transformed = [(peak[0]*_rscale, peak[1]+(_rot % (2*np.pi))) for peak in ref_peaks]  # Similarity transform
    return loss(ref_transformed, exp_peaks)


def merge_close(coordinates: list[tuple[float, float]],
                tol: float,
                show_graph: bool = False)\
        -> list[tuple[float, float]]:
    """Merge closely-neighboring points into a single point, with coordinates at the center of mass of the neighborhood.
    Args:
        coordinates: A collection of cartesian point coordinates.
        tol: The maximum Euclidean distance to count as a neighbor.
        show_graph: Whether to plot the near-neighbor graph; useful for debugging (default False)
    Returns:
        The list of coordinates after merging.
    """
    # Algorithm: 1) Get the full distance matrix (fast for a small number of points)
    #            2) For each point, find all of its neighbors
    #            3) Construct an undirected graph from the points (neighbors have edges between them)
    #            4) Find all the connected components of the graph
    #            5) Add a new point at the center of mass of each connected component
    #            6) Remove the unmerged points
    dists = scidist.squareform(scidist.pdist(np.array(coordinates),
                                             metric="euclidean"))
    close_neighbor_edges = set()
    for i in range(len(coordinates)):
        neighbors = np.argwhere(dists[i, :] < tol).flatten()
        if neighbors.size == 1:  # Disregard points which only neighbor themselves
            continue
        # Set comprehension to ignores duplicate edges; frozenset is unordered and hashable (unlike tuple or set)
        new_edges = {frozenset([i, n]) for n in neighbors if n != i}
        close_neighbor_edges.update(new_edges)
    edgelist = list(map(tuple, close_neighbor_edges))  # Format as list of tuples for networkx to consume
    graph = nx.from_edgelist(edgelist)
    if show_graph:
        nx.draw_networkx(graph, node_size=10, arrowsize=20,
                         node_color="#ffffff", edge_color="#ffffff", font_color="#0088ff")
        messagebox.showinfo(title="networkx.draw(graph)",
                            message="Displaying point connectivity graph; press OK to continue.",
                            detail="Distances in connectivity graph do not correspond to Euclidean distances.")
    connected_components = list(nx.connected_components(graph))

    merged_coords, drop_indices = [], []
    for comp in connected_components:
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

    final_coords = copy.copy(coordinates)
    for i in sorted(drop_indices, reverse=True):  # Remove unmerged points from end first
        del final_coords[i]
    final_coords.extend(merged_coords)
    return final_coords


def test_analysis_stack(analysis_stack: np.ndarray) -> None:
    global fig, axs, ax, i
    fig, axs = plt.subplots(4, 4, figsize=(4, 4),
                            subplot_kw={"xticks": [], "yticks": []},
                            gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    for ax in axs.flat:
        i = np.random.randint(len(analysis_stack))
        ax.imshow(analysis_stack[i], cmap="inferno")


# %% Open .cif files for each relevant phase
element_of_interest: str = "Hf"  # Currently only supporting single-site vPCFs
ignore_elements: list[str] = ["O"]  # The listed elements will be skipped when generating vPCF coordinates
test_refs: bool = False  # Set this to enable displaying coordinate plots (for debugging)

file_list = [Path(file) for file in
             askopenfilenames(filetypes=[("Crystallographic Information File", ".cif")])]
print([f"{i}: {f.stem}" for i, f in enumerate(file_list)])

# %% SingleOrigin setup
# The following code contains some hard-coded values, and may need to be adjusted for other structures!
fitting_file: Path = file_list[2]  # Change to select the .cif file to use for initial fitting

uc = so.UnitCell(str(fitting_file), origin_shift=(0.1, 0.1, 0))
uc.project_zone_axis((0, 1, 1),  # Zone axis direction
                     (0, 2, 0),
                     (2, 0, 0),
                     ignore_elements=ignore_elements,
                     reduce_proj_cell=False)
uc.combine_prox_cols(toler=1)

image, _ = so.load_image(path=askopenfilename(), display_image=False, images_from_stack="all")
hrimage = so.HRImage(image)
lattice = hrimage.add_lattice("lattice", uc)
lattice.get_roi_mask_polygon()

# %% Fit lattice to image, and refine
lattice.fft_get_basis_vect(a1_order=4, a2_order=4, sigma=3)
lattice.define_reference_lattice(plot_ref_lattice=False)
lattice.fit_atom_columns(buffer=10, local_thresh_factor=0.0, parallelize=True,
                         use_circ_gauss=True, watershed_line=True)
plot_fit_lattice: bool = True
if plot_fit_lattice:
    hrimage.plot_atom_column_positions(fit_or_ref='fit', outlier_disp_cutoff=100, scalebar_len_nm=None)

# %% Extract sub-images and local column coordinates for each
window_size: int = 64  # Set kernel size (px)
test_subimages: bool = True  # Whether to show a few sub-images as a sanity check

raw = image.reshape(1, image.shape[0], image.shape[1])
lattice.at_cols["class"] = 0  # Class number, required by atomai
coords = lattice.at_cols[["x_fit", "y_fit", "class"]].to_numpy()
# noinspection PyTupleAssignmentBalance
imstack, coms, _ = atomai.utils.extract_subimages(np.expand_dims(raw, axis=3),
                                                  coordinates={0: np.array(coords)},
                                                  window_size=window_size,
                                                  coord_class=0)
imstack = imstack[..., 0]  # Disregard channel dimension

if test_subimages:
    test_analysis_stack(imstack)

# %% Select what will be used for k-means cluster analysis
test_vpcfs: bool = True  # Whether to show some FFTs to verify things are working as expected

coords_around_point = []
for frame in range(imstack.shape[0]):
    given_point = coms[frame]
    # Extracting coordinates around the given point
    distances = np.sqrt((lattice.at_cols["x_fit"]-given_point[0])**2 +
                        (lattice.at_cols["y_fit"]-given_point[1])**2)
    indices = distances[distances <= window_size].index
    coords_around_point.append(lattice.at_cols.loc[indices])

lattice_copy = copy.copy(lattice)
vpcf = []
for i in range(len(coords_around_point)):
    lattice_copy.at_cols = coords_around_point[i]
    # TODO: RESULTING vPCFs MUST BE SQUARE; set this up to do that automatically
    lattice_copy.get_vpcfs(xlim=[-1, 1],
                           ylim=[-0.679, 0.679],
                           d=0.0555)
    vpcf_name = f"{element_of_interest}-{element_of_interest}"
    vpcf.append(lattice_copy.vpcfs[vpcf_name])
vpcfs = np.array(vpcf)
print(vpcfs.shape)  # Print to visually check that all vPCFs are square
analysis_stack = vpcfs[:, :-1, :-1]  # TODO: Need to re-size vPCFs to be window_size

if test_vpcfs:
    test_analysis_stack(analysis_stack)

# %% k-means clustering
# TODO: Maybe try DBSCAN?
num_clusters: int = 4  # Hyperparameter for k-means: number of clusters in image
show_kmeans_central_members: bool = False  # Whether to show representative (central) vPCFs for each k-means cluster
show_kmeans_map: bool = True  # Whether to show the spatial map of k-means clusters

# noinspection PyUnboundLocalVariable
z = analysis_stack.reshape(-1, window_size**2)
kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(z)

if show_kmeans_central_members:
    rows = int(np.ceil(float(num_clusters)/4))
    cols = int(np.ceil(float(num_clusters)/rows))
    gs = matplotlib.gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(4*cols, 4*(1+rows//1.5)))
    for i in range(num_clusters):
        ax = fig.add_subplot(gs[i])
        with warnings.catch_warnings():  # It's fine to ignore divide-by-zero warnings in this case
            warnings.filterwarnings("ignore", "divide by zero encountered in log")
            warnings.filterwarnings("ignore", "invalid value encountered in log")
            ax.imshow(np.log(kmeans.cluster_centers_[i, :].reshape(window_size, window_size)), cmap='inferno')
        ax.set_title('Cluster Center - ' + str(i + 1))
        plt.tick_params(labelsize=18)
        plt.axis('off')
    plt.show()

if show_kmeans_map:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
    ax.imshow(raw[0], cmap="gray")
    scatter = ax.scatter(coms[:, 0], coms[:, 1], c=kmeans.labels_, cmap="jet", alpha=1)
    ax.set_title('kMeans labels')
    ax.set_axis_off()
    plt.colorbar(scatter)
    plt.show()

# %% Generate total vPCFs for each cluster
# This is technically a bit different from just the central members, and should give more positional accuracy
show_total_vpcfs: bool = True  # Whether to show the total vPCFs

lattice_copy_2 = copy.copy(lattice)
hrimage_2 = copy.copy(hrimage)
labels = np.unique(kmeans.labels_)
vpcf_clusters = []
vpcf_origins = []
for i in range(labels.shape[0]):
    coor_cluster = coms[np.where(kmeans.labels_ == i)[0]]
    total_cluster = lattice.at_cols[(lattice.at_cols['x_fit'].isin(coor_cluster[:, 0]))
                                    & (lattice.at_cols['y_fit'].isin(coor_cluster[:, 1]))]
    lattice_copy_2.at_cols = total_cluster
    lattice_copy_2.get_vpcfs(xlim=[-1, 1],
                             ylim=[-0.679, 0.679],
                             d=0.0555)
    vpcf_name = f"{element_of_interest}-{element_of_interest}"
    vpcf_clusters.append(lattice_copy_2.vpcfs[vpcf_name])
    vpcf_origins.append(lattice_copy_2.vpcfs["metadata"]["origin"])

if show_total_vpcfs:
    fig, axs = plt.subplots(nrows=1, ncols=num_clusters, figsize=(4*num_clusters, 4))
    for i, clust in enumerate(vpcf_clusters):
        axs[i].imshow(clust, cmap="inferno")

# %% Generate vPCF coordinates for the perfect structures
# Tuples are (zone axis vector, basis vector 2, basis vector 3)
zones = [((0, 0, 1), (5, 0, 0), (0, 5, 0)),
         ((0, 1, 0), (5, 0, 0), (0, 0, -5)),
         ((1, 0, 0), (0, 0, -5), (0, 5, 0)),
         ((1, 1, 0), (5, -5, 0), (0, 0, -5)),
         ((1, 0, 1), (5, 0, -5), (0, -5, 0)),
         ((0, 1, 1), (0, -5, 5), (5, 0, 0)),
         ((1, 1, 1), (5, 0, -5), (-5, 10, -5))]

_temp_mapping = {(fname, axes): None for fname in file_list for axes in zones}  # TODO: May not need to be a dict?
struct_mapping = {}
for mapping in _temp_mapping.keys():
    # TODO: Compress the mapping (merge indistinguishable vPCFs under a single key) -- work in _temp_mapping?
    map_name = str((mapping[0].stem, mapping[1][0]))
    struct_mapping[map_name] = polar_vpcf_peaks_from_cif(mapping[0], mapping[1], "Hf")
del _temp_mapping

if test_refs:  # Plot one of the phase-and-zones to check if things are working
    for struct_map in struct_mapping.items():
        peaks = struct_map[1]  # Select phase-and-zone

        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        ax.set_rlim(rmin=0, rmax=np.ceil(np.max(np.array(list(peaks))[:, 0])))
        ax.set_thetagrids((0, 90, 180, 270), ("0", u"\u03c0/2", u"\u03c0", "3"+u"\u03c0/2"))
        ax.scatter(np.array(list(peaks))[:, 1], np.array(list(peaks))[:, 0])
        fig.suptitle(struct_map[0])
        fig.show()

# %% Fit modeled vPCFs to total vPCFs
# TODO: XA and XB must have the same number of columns
# We need to make  sure that we use the same number of peaks in both experimental and reference vPCFs.
# Perhaps we can sort r-theta and just take the first n peaks?
variable_name = []
for i, (vc, o) in enumerate(zip(vpcf_clusters, vpcf_origins)):
    optimized = {}
    exp_peaks = [xy_to_rt(xy) for xy in extract_peak_coords(o, vc)]
    for key, ref_peaks in struct_mapping.items():
        print(f"Optimizing fit between experimental vPCF {i} and reference vPCF {key}...")
        optimized[key] = optimize.minimize(optimize_wrapper, np.array([1, 0]), args=(ref_peaks, exp_peaks))
    variable_name.append(optimized)

#%%
