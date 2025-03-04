from pymatgen.io import cif
from tkinter.filedialog import askopenfilename, asksaveasfilename
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from warnings import warn
import matplotlib
from tkinter import Tk
import numpy as np


def tk_popover(save: bool = False, **kwargs):
    """Tk helper to ensure window appears on top."""
    root = Tk()
    root.iconify()
    root.attributes('-topmost', True)
    root.update()
    loc = None  # Default return if open fails; will likely cause an error when passed along
    try:
        if not save:
            loc = askopenfilename(parent=root, **kwargs)
        else:
            loc = asksaveasfilename(parent=root, **kwargs)

    finally:
        root.attributes('-topmost', False)
        root.destroy()
    return loc


# %% Read & tile
parser = cif.CifParser(tk_popover())
struct = parser.parse_structures(primitive=False)[0]
scaling_matrix = np.array([[3, 0, 0],  # User-defined supercell tiling
                           [0, 3, 0],
                           [0, 0, 3]])
struct.make_supercell(scaling_matrix)

# %% Get convex hull
allcoords = np.array([s.coords for s in struct.sites])
if len(allcoords) > 4000:
    warn("FEFF has a hard-coded limit of 4000 atoms when using the ATOMS card.")
hull = ConvexHull(allcoords)
hull_eqs = hull.equations
norms = np.linalg.norm(hull_eqs[:, :-1], axis=1, keepdims=True)
distances = np.abs(hull_eqs[:, :-1] @ allcoords.T + hull_eqs[:, -1:]) / norms
min_distances = distances.min(axis=0)

# %% Set FMS radius
rfms: float = 8  # Angstrom

# %% Visualize and confirm that it looks plausible
# Note: this is just for sanity chacking; the color-coded interior atoms are not filtered by element
matplotlib.use('TkAgg')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
exterior_atoms = allcoords[min_distances < rfms]
interior_atoms = allcoords[min_distances > rfms]
ax.scatter(exterior_atoms[:, 0], exterior_atoms[:, 1], exterior_atoms[:, 2],
           color='blue', s=10, label='Exterior Atoms')
ax.scatter(interior_atoms[:, 0], interior_atoms[:, 1], interior_atoms[:, 2],
           color='red', s=10, label='Interior Atoms')
for simplex in hull.simplices:
    vertices = allcoords[simplex]
    hull_face = Poly3DCollection([vertices], alpha=0.2, color='green', edgecolor='k')
    ax.add_collection3d(hull_face)
ax.legend()
plt.ion()
plt.show()

# %% Output
# FEFF needs to label each position with its potential index (defined in feff.inp)
# Define the mapping here, just make sure it's consistent with the POTENTIALS card
# Whichever element is potential index 1 will have its first (interior) instance labeled as potential 0 (ELNES target)
# All interior instances of the element with potnential index 1 will be labeled with a comment in the output
potential_mapping = {"N": 1,
                     "Al": 2,
                     "Gd": 3}

first = True
with open(tk_popover(save=True), "wt") as outfile:
    for dist, site in zip(min_distances, struct.sites):
        outfile.write(f"{site.x:.7f}\t{site.y:.7f}\t{site.z:.7f}\t"
                      f"{potential_mapping[site.species_string]}\t{site.species_string}"
                      f"{' * interior' if dist > rfms and potential_mapping[site.species_string]==1 else ''}\n")

#%%
