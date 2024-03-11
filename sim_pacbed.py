# %%
"""PACBED Thickness Series Simulation

This version of the PACBED image simulation script is designed to be used with abTEM versions >= 1.0.0. This script
generates PACBED images based on an input .cif file, and automatically creates a thickness series of images. Commonly
adjusted settings are documented in the Parameters dataclass.

The basic logic of the script is:
    0.  User sets the simulation parameters
    1.  Read the structure from a .cif file (locally or automatically downloaded from the Materials Project)
    2.  Initialize a pixelated detector to caputre the PACBEDs
    3.  Generate a full-thickness model from the .cif file
    4.  If using frozen phonons, generate the configurations
    5.  Create a potential object representing all frozen phonon configurations
    6.  Define the STEM probe and a well-sampled (Nyquist) scan over a single projected cell
    7.  Run the computation!
    8.  Sum over the scan axes of the resulting array to convert CBED -> PACBED
    9.  Export the PACBEDs as a .tif stack
    10. Repeat steps 3-9 for all specified zone axes
"""
import abtem
import dask
import numpy as np
from ase import io, build
from ase.visualize import view
from dataclasses import dataclass
from packaging import version
from pathlib import Path
from platform import system
from tifffile import imwrite
from tkinter.filedialog import askopenfilename, askdirectory
from typing import Literal
from warnings import warn

if not version.parse(abtem.__version__) >= version.parse("1.0.0"):
    raise RuntimeError("This script will only work with abTEM versions >= 1.0.0 due to breaking changes in abTEM. "
                       "Please update you abTEM version.")


# %%
@dataclass
class PACBEDPrms:
    """Parameters for the PACBED simulation.  Each parameter has its own documentation."""

    device: Literal["gpu", "cpu"]
    """Whether to run the simulation on the GPU or the CPU."""

    beam_energy: float
    """Simulated electron beam energy (electronvolts)."""

    convergence: float
    """Semiangle of convergence of the electron probe (miliradians)."""

    sampling: float
    """Sampling of the model potential (Angstrom). Smaller values will increase the maximum frequency which can be 
    represented in reciprocal space via the equation ``|k| < min(Nx/(2*a), Ny/(2*b))`` where ``Nx`` and ``Ny`` are 
    the number of samples in the x and y directions (``a`` and ``b`` are the model dimensions; see ``tiling`` below). 
    Samller samplings also result in longer simulation times. A uniform resampling is applied to all PACBEDs to 
    ensure square pixels even for non-square models; as a result non-square models will produce unequal numbers 
    of pixels in the kx and ky dimensions."""

    tiling: int | tuple[int, int] | list[int] | list[tuple[int, int]]
    """Number of times to tile the projected cell; more tiling will give higher resoution PACBEDs at the cost of 
    simulation speed and memory usage. If a single integer is passed then tiling will be the same in both x and y 
    directions; if different tilings are desired in x and y (e.g. for non-square projected cells) then a tuple can be 
    passed instead. If multiple zone axes will be simulated (see ``zone`` below) then a list of either ints or tuples 
    can be passed and will be applied to each zone axis in the order they are specified."""

    zone: tuple[int, int, int] | list[tuple[int, int, int]]
    """Zone axes along which to project the structure and simulate PACBEDs, in the form ``(h, k, l)``. If a 
    list is provided, a series of PACBEDs will be simulated and saved for each zone axis in the list."""

    max_angle: float
    """The maximum angle to be detected when forming PACBEDs. Must be strictly less than the maximum representable 
    frequency (see ``sampling`` for details). Mainly used to crop out large areas of near-zero intensity from 
    the simulated PACBEDs when high convergence angles are used."""

    thickness: float
    """Total model thickness (Angstrom). Will be rounded to the nearest whole (projected) unit cell."""

    thickness_step: float
    """Thickness step between each returned PACBED (Angstrom), e.g. a model with ``thickness=100`` and 
    ``thickness_step=20`` will return 5 PACBEDs with each representing an additional 20A of model thickness."""

    slice_thickness: float
    """Thickness of each slice used in the multislice simulation algorithm (Angstrom). If possible try to avoid values 
    which will result in atoms being sliced exactly in half, e.g. if the projected cell is 4A thick with an atom at 
    the center then ``slice_thickness=2`` results in ambiguous behavior which may affect the appearance of the final 
    PACBED. Optimal value depends on the specifics of the particular model."""

    phonon_configs: int
    """The number of frozen phonon configurations to be used during simulation. Generally only a few are needed for 
    good qualitative resutls. Set to 0 to disable frozen phonons and use a "perfect" model without thermal effects."""

    phonon_sigmas: dict[str, float]
    """The sigma values for the frozen phonon configurations (Angstrom), where ``sigma = sqrt(U_iso)``. You can 
    convert from B_iso by dividing by a constant geometric factor: ``U_iso == B_iso / (8*pi**2)``. Should be 
    supplied as a dictionary from elemental symbol to sigma value, as:
        phonon_sigmas = {"A": 0.01,
                         "B": 0.02,
                         "X": 0.15}
    If ``phonon_configs > 0`` then sigmas must be supplied for _all_ elements present in the model."""

    seed: int | None
    """Seed for generation frozen phonon configurations. Set to ``None`` to automatically generate a new seed each time
    the simulation is run."""

    tilt_mag: float
    """Sample mistilt magnitude (miliradians). This is approximated via a tilt of the electron probe and so will 
    only be accurate for small mistilts. For large mistilts the model should be rotated."""

    tilt_angle: float
    """The angle of mistilt (counterclockwise from the +x direction, radians)."""

    def __post_init__(self):  # This section sanity-checks the various parameters and ensures consistency
        if self.device not in ["gpu", "cpu"]:
            raise RuntimeError("Device must be one of ``'gpu'`` or ``'cpu'``.")

        for prm in [("beam_energy", self.beam_energy),
                    ("convergence", self.convergence),
                    ("sampling", self.sampling),
                    ("max_angle", self.max_angle),
                    ("thickness", self.thickness),
                    ("thickness_step", self.thickness_step),
                    ("slice_thickness", self.slice_thickness),
                    ("phonon_configs", self.phonon_configs),
                    ("tilt_mag", self.tilt_mag)
                    ]:
            # noinspection PyTypeChecker
            if not prm[1] >= 0:
                raise RuntimeError(f"``{prm[0]}`` must be non-negative.")

        if not self.thickness_step <= self.thickness:
            raise RuntimeError("``thickness_step`` must be <= ``total thickness``.")
        if not self.slice_thickness <= self.thickness_step:
            raise RuntimeError("``slice_thickness`` muse be <= ``thickness_step``.")

        if type(self.zone) is tuple:
            if not (type(self.tiling) is int or type(self.tiling) is tuple):
                raise RuntimeError("``tiling`` cannot be a list if ``zone`` is not.")
        if type(self.zone) is list and type(self.tiling) is list:
            if not len(self.zone) == len(self.tiling):
                raise RuntimeError("Length mismatch between ``zone`` and ``tiling``.")
        if type(self.tiling) is int:
            if not self.tiling > 0:
                raise RuntimeError("``tiling`` must be positive and non-zero.")
        if type(self.tiling) is tuple:
            if not len(self.tiling) == 2:
                raise RuntimeError("``tiling`` must have two dimensions.")
            for t in self.tiling:
                if not type(t) is int:
                    raise RuntimeError("``tiling`` must contain only integers.")
                if not t > 0:
                    raise RuntimeError("All ``tiling`` dimesions must be positive and non-zero.")
        if type(self.tiling) is list:
            for til in self.tiling:
                if type(til) is int:
                    if not til > 0:
                        raise RuntimeError("All ``tiling``s must be positive and non-zero.")
                if type(til) is tuple:
                    if not len(til) == 2:
                        raise RuntimeError("``tiling`` must have two dimensions.")
                    for t in til:
                        if not type(t) is int:
                            raise RuntimeError("All ``tiling``s must contain only integers.")
                        if not t > 0:
                            raise RuntimeError("All ``tiling`` dimensions must be positive and non-zero.")

        if self.beam_energy < 10E3:
            warn(f"Beam energy is very low ({self.beam_energy}eV). Are you sure this is what you meant to do?")
        if self.convergence > 100:
            warn(f"Convergence angle is very high ({self.convergence}mrad). Are you sure this is what you meant to do?")
        if self.phonon_configs > 0:
            known_elements = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
                              "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
                              "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                              "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
                              "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
                              "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
                              "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
                              "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"}
            phonon_keys = self.phonon_sigmas.keys()
            for key in phonon_keys:
                if key not in known_elements:
                    warn(f"One or more key in phonon_configs is not a known element ({key}). Are you sure this is what"
                         " you meant to do?")


# noinspection PyShadowingNames
def _what_resolution(a: float,
                     b: float,
                     t: int | tuple[int, int])\
        -> float:
    """Helper function to determine final resolution (written to tif metadata)

    Uniform resampling of the detector downsamples the pixel size of the final PACBEDS, resulting in square pixels
    with whatever the larger dimension was originally

    Args:
        a: Dimension a (x) of the projected cell (Angstrom)
        b: Dimension b (y) of the projected cell (Angstrom)
        t: The tiling of the projected cell, which might be isotropic (same for a and b) or anisotropic

    Returns: The larger of the two pixel sizes, i.e. the resulting pixel size of the PACBED (1/Angstrom)
    """
    if type(t) is int:  # Isotropic tiling
        a_res = 1/(a*t)
        b_res = 1/(b*t)
    elif type(t) is tuple:  # Anisotropic tiling
        a_res = 1/(a*t[0])
        b_res = 1/(b*t[1])
    else:  # Something went wrong, return default value
        a_res = b_res = 0

    return max(a_res, b_res)


def _minmax_norm(img):
    _min, _max = np.min(img), np.max(img)
    normed = (img - _min) / (_max - _min)
    return normed


# %% Define simulation parameters
# All parameters are required
prms = PACBEDPrms(
        device="gpu",
        beam_energy=200E3,
        convergence=17.9,
        sampling=0.1,
        zone=(1, 1, 0),
        tiling=(27, 30),
        max_angle=50,
        thickness=200,
        thickness_step=200,
        slice_thickness=2,
        phonon_configs=0,
        phonon_sigmas={"Al": 0.0567,  # DOI: 10.1107/S0108767309004966
                       "Sc": 0.0567,  # No good source for this, assuming the same as Al
                       "N":  0.0593},
        seed=None,
        tilt_mag=50,
        tilt_angle=float(np.radians(90)))  # Auto-convert degrees to radians

# %% Low-level configuration settings
# These only need to be adjusted in the case of out-of-memory errors or very poor performance
abtem.config.set({"device": prms.device,
                  "dask.lazy": True,  # Setting to False can be useful for debugging
                  "dask.chunk-size": "128 MB",  # Standard L3 cache size (per core)
                  "dask.chunk-size-gpu": "4096 MB"})  # Remember to leave space for overhead

if prms.device == "gpu":  # If running on the GPU, Dask needs to be configured
    if system() == "Linux":  # If we're on Linux we can use multiple GPUs
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        cluster = LocalCUDACluster()
        client = Client(cluster)  # abTEM automatically sets Dask to distributed scheduling if a client is instantiated
    else:  # Fall back to single-GPU execution
        dask.config.set({"num_workers": 1})  # Single GPU execution

# %% Load Structure
# Note: ASE supports formats other than .cif; if you would like to use another format see their documentation at
#       https://wiki.fysik.dtu.dk/ase/ase/io/io.html
load_method: Literal[
    "fixed",        # Load via a fixed file path, given below, without a file picker window
    "interactive",  # Interactively select the file using a (native) file picker window
    "mp_api"        # Load a cif from Materials Project; requires an API key!
    ] = "fixed"

fixed_path = Path("E:/Users/Charles/AlN.cif")  # Local path to .cif file
mp_id = "mp-661"  # Materials Project ID for desired structure, including the leading "mp-"

match load_method:
    case "fixed":
        struct = io.read(fixed_path)
        fname = fixed_path.stem
    case "interactive":
        interactive_path = Path(askopenfilename())
        struct = io.read(interactive_path)
        fname = interactive_path.stem
    case "mp_api":
        from mp_api.client import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor
        from data_analysis_scripts.cse_secrets import MP_API_KEY
        mp_api_key = MP_API_KEY  # Personal API key (tied to MP account)

        with MPRester(mp_api_key) as mp:
            mp_struct = mp.get_structure_by_material_id(mp_id)
            print(mp_struct)
            struct = AseAtomsAdaptor().get_atoms(mp_struct)
        fname = mp_id
    case _:
        raise NotImplementedError(f"Unrecognized file load method: {load_method}")

# %% View the structure to confirm everything looks right
# Opens 3D model in an interactive window
view(struct)

# %% Define the save location
savedir = Path(askdirectory())
if savedir == Path("."):  # Returned root
    warn("Save directory is currently set to root ('.').  Are you sure this is what you meant to do?")

# %% Run the simulation
# noinspection PyTypeChecker
detector = abtem.PixelatedDetector(max_angle=prms.max_angle,
                                   reciprocal_space=True,
                                   resample="uniform")

if type(prms.zone) is not list:
    prms.zone = [prms.zone]  # Wrap in list for simpler syntax below
if type(prms.tiling) is not list:  # Means we were only passed a single tiling
    n_zones = len(prms.zone)
    prms.tiling = [prms.tiling] * n_zones

for za, tile in zip(prms.zone, prms.tiling):
    print(f"Beginning simulation for {za} zone axis...")

    # Temporary surface used to calculate parameters for the projected cell
    _tmp_atoms = build.surface(struct, indices=za, layers=1, periodic=True)
    a, b, c = _tmp_atoms.cell
    a = a[0]
    b = b[1]
    c = c[2]
    del _tmp_atoms  # Don't accidentally use the temporary surface

    thickness_multiplier = int(prms.thickness // c) + 1
    atoms = build.surface(struct, indices=za, layers=thickness_multiplier, periodic=True)
    atoms = abtem.orthogonalize_cell(atoms)
    if type(tile) is int:
        atoms *= (tile, tile, 1)
    elif type(tile) is tuple:
        atoms *= (tile[0], tile[1], 1)
    view(atoms)

    if prms.phonon_configs != 0:
        configs = abtem.FrozenPhonons(atoms,
                                      sigmas=prms.phonon_sigmas,
                                      num_configs=prms.phonon_configs,
                                      seed=prms.seed)
    else:
        configs = abtem.FrozenPhonons(atoms,
                                      sigmas=0,  # No displacement ==> disables frozen phonons
                                      num_configs=1)

    save_delta = round(prms.thickness_step / prms.slice_thickness)
    potential = abtem.Potential(configs,
                                sampling=prms.sampling,
                                projection="infinite",
                                parametrization="kirkland",
                                slice_thickness=prms.slice_thickness,
                                exit_planes=save_delta)

    tilt = (prms.tilt_mag * np.cos(prms.tilt_angle),  # x
            prms.tilt_mag * np.sin(prms.tilt_angle))  # y
    probe = abtem.Probe(semiangle_cutoff=prms.convergence,
                        extent=potential.extent,
                        sampling=prms.sampling,
                        energy=prms.beam_energy,
                        tilt=tilt)
    probe.match_grid(potential)  # With above settings the grids should already match, but just in case

    scan = abtem.GridScan(start=(0., 0.),
                          end=(a, b),  # Only need to scan one projected cell to get a complete PACBED
                          sampling=probe.aperture.nyquist_sampling)

    measurement = probe.scan(potential=potential,
                             scan=scan,
                             detectors=detector)
    measurement.compute()

    # pacbed_stack axis order is: Z X Y Kx Ky
    pacbed_stack = np.sum(measurement.array, axis=(1, 2))
    for i, plane in enumerate(pacbed_stack):
        pacbed_stack[i] = _minmax_norm(plane)
    pacbed_stack = pacbed_stack.astype("float32")  # Z Kx Ky

    print("Exporting...", end=" ")

    if "fname" not in locals():
        # Normal program flow should not be able to get here, but just in case we somehow do: set a default fname
        # so that whatever we just simulated gets saved (better to back up bad data than discard good data)
        fname = "DEFAULT"
        warn("Cannot infer file name, setting to 'DEFAULT'. Recommended to check results: simulation may have "
             "been in an unexpected state.")

    # Information redundancy in both filename and tiff description field (in case filename is changed)
    export_name = f"{fname}_PACBED_tilt{prms.tilt_mag}mrad@{np.degrees(prms.tilt_angle)}deg_" +\
                  f"{str(za)}_{str(int(prms.thickness))}A_with_stepsize_{str(int(prms.thickness_step))}A.tif"
    res = _what_resolution(a, b, tile)
    description = f"name: {fname}\n" \
                  f"zone: {za}\n" \
                  f"thickness: {prms.thickness} A\n" \
                  f"step: {prms.thickness_step} A\n" \
                  f"tilt: {prms.tilt_mag}mrad@{np.degrees(prms.tilt_angle):.1f}deg\n" \
                  f"resolution: {res:0.4f} A^-1/px"

    slice_labels = [f"{tks:.1f} A" for tks in potential.exit_thicknesses]
    imwrite(savedir / export_name,
            pacbed_stack,
            photometric="minisblack",
            software="abTEM",
            metadata={"axes": "ZYX",
                      "Labels": slice_labels,
                      "Info": description})

    print("Done!")

print("Simulation complete!")

#%%
