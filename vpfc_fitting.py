from tkinter.filedialog import askopenfilenames
from pathlib import Path
import numpy as np
import SingleOrigin as so
import ase.io as aio
import ase.build as abuild
import matplotlib
import matplotlib.pyplot as plt
from abtem.atoms import orthogonalize_cell

matplotlib.use("TkAgg")
plt.style.use('dark_background')

# %% Open .cif files for each relevant phase
file_list = [Path(file) for file in
             askopenfilenames(filetypes=[("Crystallographic Information File", ".cif")])]
print([f"{i}: {f.stem}" for i, f in enumerate(file_list)])

# %% Provide a list of candidate zone axes
# The vPCF parameters of each zone axis will be computed for each phase,
# so for n phases and m zone axes there will be n*m possible vPCFs to test against
zas = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
radius = 10  # The model radius for generating vPCF parameters (Angstrom)
skip_elements: list[str] = ["O"]  # The listed elements will be skipped when generating the test vPCF parameters
test: bool = True  # Set this flag to show the first structure in structural_mapping and then break; for debugging

structural_mapping = {(fname, za): None for fname in file_list for za in zas}
for mapping in structural_mapping.keys():
    # Build the test surface
    test_cell = aio.read(mapping[0])
    # test_cell.rotate(mapping[1], "z", rotate_cell=True)
    if test:
        from ase import visualize
        visualize.view(test_cell * (4, 4, 4))
        break
    span = {"a": np.linalg.norm(test_cell.cell[0]),
            "b": np.linalg.norm(test_cell.cell[1])}
    mul = (int(np.ceil(2*radius/span["a"])), int(np.ceil(2*radius/span["b"])), 1)
    test_cell *= mul
    # Drop unwanted atoms and select unique x-y positions
    selected_positions = [pos for pos, e in zip(test_cell.get_positions(), test_cell.get_chemical_symbols())
                          if e not in skip_elements]
    roundoff = 3  # Need to round off to some precision for set reduction to work well; 0.1 pm seems fine
    unique_xy = {(round(pos[0], roundoff),
                  round(pos[1], roundoff))
                 for pos in selected_positions}
    # Center coordinates and convert to polar
    center = (np.mean([xy[0] for xy in unique_xy]), np.mean([xy[1] for xy in unique_xy]))
    unique_xy = {(xy[0]-center[0], xy[1]-center[1]) for xy in unique_xy}
    unique_polar = {(np.linalg.norm((x, y)), np.arctan(y/x)) if x > 0
                    else (np.linalg.norm((x, y)), np.arctan(y/x)+np.pi)  # For points in quadrants II or III
                    for (x, y) in unique_xy}
    structural_mapping[mapping] = unique_polar

# %% Check that everything looks reasonable
test_polar = next(iter(structural_mapping.values()))
test_xy = {(r*np.cos(t), r*np.sin(t)) for r, t in test_polar}

fig = plt.figure()
ax = fig.add_subplot(projection="polar")
plt.rgrids((5, 10, 15))
plt.thetagrids((0, 90, 180, 270), ("0", u"\u03c0/2", u"\u03c0", "3"+u"\u03c0/2"))
ax.scatter(np.array(list(test_polar))[:, 1], np.array(list(test_polar))[:, 0])
fig.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.set_aspect("equal")
ax2.scatter(np.array(list(test_xy))[:, 0], np.array(list(test_xy))[:, 1])
fig2.show()

# %% Pick one .cif for initial fitting and create a so.UnitCell object from it
fitting_file: Path = file_list[2]  # Change slice to select the .cif file to use for initial fitting
plot_projected_cell: bool = True
rough_tol: float = 1e0  # Tolerance for combining atoms into a single column (Angstrom)

uc = so.UnitCell(str(fitting_file), origin_shift=(0.1, 0.1, 0))
uc.project_zone_axis((0, 1, 1),  # Zone axis direction
                     (0, 2, 0),  # Most horizontal axis in projection
                     (2, 0, 0),  # Most vertical axis in projection
                     ignore_elements=skip_elements,
                     reduce_proj_cell=False)
uc.combine_prox_cols(toler=rough_tol)
if plot_projected_cell:
    uc.plot_unit_cell()

#%%
