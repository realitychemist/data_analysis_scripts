"""Spatial statistics analysis of EDXS data, to formally test against CSR (Poisson noise)."""
# Data handling and analysis
from data_analysis_scripts.utils import tk_popover
import hyperspy.api as hs
import numpy as np
from scipy import stats
from libpysal import weights
from esda import Moran_Local, fdr

# Plotting & display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from shapely.geometry import box
from geopandas import GeoDataFrame
from tabulate import tabulate

matplotlib.use("TkAgg")

# %% Load the .emd
infile = hs.load(tk_popover(),
                 load_SI_image_stack=True,
                 sum_frames=False,
                 lazy=False,
                 rebin_energy=8,
                 SI_dtype="uint8")

# %% Get overall spectrum
whole_dataset = None
for f in infile:
    if "EDSTEMSpectrum" in str(f.T.T):
        whole_dataset = f
if whole_dataset is None:
    raise RuntimeError("Failed to find full dataset")

detectors = whole_dataset.original_metadata["Detectors"]
dispersion = 1  # eV – default value in case it cannot be found in the file, almost certainly wrong
for det in detectors:
    try:
        if "SuperXG2" in det[1].DetectorName:
            dispersion = int(det[1].Dispersion)  # eV
            break
    except AttributeError:  # Non-EDXS detectors have a differently-named field
        continue

integrated_spectrum = np.sum(np.sum(whole_dataset.data, axis=0), axis=0)
# Shift to align strobe peak to zero
# Note: Below depends on strobe peak being larger than any signal peaks; _usually_ true, but not guaranteed
strobe_idx = np.argmax(integrated_spectrum)
energy_axis = list(np.arange(start=dispersion*strobe_idx*-1,
                             stop=(dispersion*len(integrated_spectrum))-(dispersion*strobe_idx),
                             step=dispersion))
energy_axis = [e/1000 for e in energy_axis]

plt.plot(energy_axis[strobe_idx+20:], integrated_spectrum[strobe_idx+20:],
         linewidth=0.5, color="black")

# These are specific to the experiment I developed this scrip for
# Use this as an example if looking at different elements
# plt.text(1.486, max(integrated_spectrum[np.argwhere([1.4 < e < 1.5 for e in energy_axis])])*1.05,
#          "Al Kα", horizontalalignment="center")
# plt.text(4.088, max(integrated_spectrum[np.argwhere([4.0 < e < 4.1 for e in energy_axis])])*1.1,
#          "Sc Kα", horizontalalignment="center")
# plt.text(4.461, max(integrated_spectrum[np.argwhere([4.4 < e < 4.5 for e in energy_axis])])*1.4,
#          "Sc Kβ", horizontalalignment="left")

plt.text(9.712, max(integrated_spectrum[np.argwhere([9.7 < e < 9.8 for e in energy_axis])])*1.05,
         "Au Lα", horizontalalignment="center")
plt.text(9.441, max(integrated_spectrum[np.argwhere([9.4 < e < 9.5 for e in energy_axis])])*1.05,
         "Pt Lα", horizontalalignment="right")
plt.text(2.120, max(integrated_spectrum[np.argwhere([2.1 < e < 2.2 for e in energy_axis])])*1.05,
         "Au M", horizontalalignment="center")
plt.text(2.048, max(integrated_spectrum[np.argwhere([2.0 < e < 2.07 for e in energy_axis])])*1.05,
         "Pt M", horizontalalignment="right")

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylabel("Counts")
plt.xlabel("Energy (keV)")
plt.show()

# %% Extract and show spectrum image
element: str = "Au"  # Element of interest
sb_size_nm: float = 20  # Scalebar size (nm)

signal = None
for f in infile:
    if "Signal2D" in str(f.T.T) and element in str(f.T.T):
        signal = f
if signal is None:
    print(f"Failed to find spectrum image for element {element}")
data = signal.data

px_size_pm = float(signal.original_metadata.BinaryResult.PixelSize["width"])  # meters
px_size_pm = px_size_pm / 10e-12  # picometers
sb_size_px = sb_size_nm / px_size_pm * 1000

datamax = int(np.max(data))
cmap = plt.get_cmap("inferno", lut=datamax+1)
norm = BoundaryNorm(np.arange(datamax+2), cmap.N)
plt.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
cbar = plt.colorbar(location="left", pad=0.01, label="Counts", ticks=np.arange(datamax+1)+0.5)
cbar.ax.set_yticklabels(np.arange(datamax+1))
cbar.ax.tick_params(size=0)

# data = data * 100  # Put on % scale
# plt.imshow(data, cmap="inferno", interpolation="nearest")
# plt.colorbar(location="left", pad=0.01, label="% Au")  # Change label for other datasets!

scalebar = AnchoredSizeBar(plt.gca().transData, sb_size_px, f"{sb_size_nm} nm", "lower left", pad=0.25,
                           color="white", frameon=False, size_vertical=2, fontproperties=FontProperties(weight="bold"))
plt.gca().add_artist(scalebar)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

# %% Kernel density estimate
kde_bandwidth: float or None = 0.02  # KDE bandwidth; if None, calculate automatically with Scott rule

y_idxs, x_idxs = np.nonzero(data)
counts = data[y_idxs, x_idxs]
positions = np.vstack([x_idxs, y_idxs])

kde = stats.gaussian_kde(positions, weights=counts, bw_method=kde_bandwidth)

xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
grid_coords = np.vstack([xx.ravel(), yy.ravel()])
kde_values = kde(grid_coords).reshape(data.shape)

plt.imshow(kde_values, cmap="inferno")
scalebar = AnchoredSizeBar(plt.gca().transData, sb_size_px, f"{sb_size_nm} nm", "lower left", pad=0.25,
                           color="white", frameon=False, size_vertical=2, fontproperties=FontProperties(weight="bold"))
plt.gca().add_artist(scalebar)
plt.colorbar(location="left", pad=0.01)

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

# %% Quadrat counts
divs: list[int] = [8, 16, 32, 64, 128, 256, 512]
if len(divs) != 7:  # Update this check if length of above list changes, but only after fixing plotting!
    raise UserWarning("The plotting code below assumes 7 sublpots, and will need to be adjusted")

signal = None
for f in infile:
    if "Signal2D" in str(f.T.T) and element in str(f.T.T):
        signal = f
if signal is None:
    print(f"Faild to find spectrum image for element {element}")

# Compute the statistics for each number of quadrats
scales_nm, quadcounts, vmrs, chi2s, pvals = [], [], [], [], []
for div in divs:
    data = signal.data  # Reload so we're not trimming the data repeatedly
    # Trim array so quadrats all have the same size
    data = data[:(data.shape[0]-(data.shape[0] % div)), :(data.shape[1]-(data.shape[1] % div))]
    quadrats = data.reshape(div, data.shape[0]//div,
                            div, data.shape[1]//div).swapaxes(1, 2)

    scales_nm.append(np.mean([quadrats.shape[-1], quadrats.shape[-2]]) * px_size_pm / 1000)

    qcounts = quadrats.sum(axis=(-1, -2))
    quadcounts.append(qcounts)

    vmrs.append(np.var(qcounts)/np.mean(qcounts))
    test_result = stats.chisquare(qcounts, axis=None)
    chi2s.append(test_result.statistic)
    pvals.append(test_result.pvalue)

tabdata = list(zip([d**2 for d in divs], scales_nm, vmrs, chi2s, pvals))
print(tabulate(tabdata, headers=["N", "Scale (nm)", "VMR", "χ²", "p"], floatfmt=".2g"))

# Plotting
fig = plt.figure()
# The gridspec and axs layout will need to be manually adjusted if a different number of divs is used!
gs = gridspec.GridSpec(2, 9, height_ratios=[1, 1])
axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
       fig.add_subplot(gs[0, 6:8]), fig.add_subplot(gs[1, 1:3]), fig.add_subplot(gs[1, 3:5]),
       fig.add_subplot(gs[1, 5:7])]
for i, ax in enumerate(axs):
    ax.imshow(quadcounts[i], cmap="inferno", interpolation="nearest")
    ax.set_title(f"{scales_nm[i]:0.2g} nm")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    bbox = ax.get_position()
fig.tight_layout()
fig.show()

# %% Sub-region analysis
# Note: must adjust subregion slicing manually if number of divs is changed, or if you want to use different subregions!
div_idx: int = 3

subregions = [quadcounts[div_idx][i:i+10, :] for i in range(0, quadcounts[div_idx].shape[0], 10)]
for i, subreg in enumerate(subregions):
    test_result = stats.chisquare(subreg, axis=None)
    print(f"Subregion {i+1}")
    print(tabulate(list(zip([np.var(subreg) / np.mean(subreg)], [test_result.statistic], [test_result.pvalue])),
                   headers=["VMR", "x2", "p"], floatfmt=".2g"))

# %% Compute local Moran stats
neighborhood: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1),    # Rook part
                                       (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Bishop part
perms: int = 10_000  # Number of permutations for testing against CSR
dataset: np.ndarray = data  # data or quadcounts[int] (for unbinned data or some binning, respectively)

local_stats = []
adjlist = {}
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[1]):
        nidxs = []
        for di, dj in neighborhood:
            if 0 <= i+di < dataset.shape[0] and 0 <= j+dj < dataset.shape[1]:
                nidxs.append((i+di, j+dj))
        adjlist[(i, j)] = nidxs
w = weights.W(adjlist)
w.transform = "r"  # Row-standard transform
ml = Moran_Local(dataset, w, permutations=perms, n_jobs=1)  # Parallelize computation

# For plotting
polygons = []
counts = []
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[1]):
        poly = box(j, -i-1, j+1, -i)  # Axes flipped for proper plotting
        polygons.append(poly)
        counts.append(dataset[i, j])
gdf = GeoDataFrame({"counts": counts, "geometry": polygons})

# %% Generate the plot
alpha: float = 0.1  # Alpha level for FDR control
quadrant: None | int = 1  # Setting this to an integer (1--4) will highlight the respective Moran quadrant:
# 1 --> High-High clusters
# 2 --> Low-High outliers
# 3 --> Low-Low clusters
# 4 --> High-Low outliers
# None --> no highlighting
ml_axs = ml.plot_combination(gdf=gdf, attribute="counts", crit_value=fdr(ml.p_sim, alpha=alpha),
                             scheme="EqualInterval", cmap="inferno", quadrant=quadrant)
plt.show()

#%%
