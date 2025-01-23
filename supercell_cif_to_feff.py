from pymatgen.io import cif
from tkinter.filedialog import askopenfilename, asksaveasfilename
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
struct = parser.parse_structures()[0]
scaling_matrix = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
struct.make_supercell(scaling_matrix)

# %% Get convex hull
allcoords = np.array([s.coords for s in struct.sites])
hull = ConvexHull(allcoords)
hull_eqs = hull.equations
norms = np.linalg.norm(hull_eqs[:, :-1], axis=1, keepdims=True)
distances = np.abs(hull_eqs[:, :-1] @ allcoords.T + hull_eqs[:, -1:]) / norms
min_distances = distances.min(axis=0)

# %% Visualize and confirm that it looks plausible
matplotlib.use('TkAgg')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
out_points = allcoords[min_distances < 8]
in_points = allcoords[min_distances >= 8]
ax.scatter(out_points[:, 0], out_points[:, 1], out_points[:, 2],
           color='blue', s=10, label='Exterior Points')
ax.scatter(in_points[:, 0], in_points[:, 1], in_points[:, 2],
           color='red', s=10, label='Interior Points')
for simplex in hull.simplices:
    vertices = allcoords[simplex]
    hull_face = Poly3DCollection([vertices], alpha=0.2, color='green', edgecolor='k')
    ax.add_collection3d(hull_face)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.ion()
plt.show()

# %% Output
first = True
with open(tk_popover(save=True), "wt") as outfile:
    for dist, site in zip(min_distances, struct.sites):
        if site.species_string == "N":
            if dist >= 8:
                interior_n = True
            else:
                interior_n = False
            if first and interior_n:  # Default to first interior N atom being the absorber; change in feff.inp file
                potential_index = 0
                first = False
            else:
                potential_index = 1
        elif site.species_string == "Al":
            interior_n = False
            potential_index = 2
        else:  # site.species_string == "Gd"
            interior_n = False
            potential_index = 3
        outfile.write(f"{site.x:.7f}\t{site.y:.7f}\t{site.z:.7f}\t{potential_index}\t{site.species_string}"
                      f"{' * interior' if interior_n else ''}\n")

#%%
