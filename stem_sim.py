from dataclasses import dataclass
from pathlib import Path
from tkinter.filedialog import askopenfilename, askdirectory
from typing import Literal
from warnings import warn

from abtem import AnnularDetector, FlexibleAnnularDetector, SegmentedDetector, PixelatedDetector
from ase import io


@dataclass
class Parameters:
    """Parameters for the PACBED simulation.  Each parameter has its own documentation."""

    device: Literal["gpu", "cpu"]
    """Whether to run the simulation on the GPU or the CPU."""

    gpu_number: int or None
    """Which GPU to use, if there are multiple available (if only one GPU is available, set to 0). Set to ``None`` if 
    running on the CPU."""

    max_batch: int
    """The maximum number of probe positions to propagate at once. Higher numbers result in faster simulations, 
    but with diminishing returns. To high of a value may actually result in slower simulations. Experimentation may 
    be required to find a good value."""

    beam_energy: float
    """Simulated electron beam energy (electronvolts)."""

    convergence: float
    """Semiangle of convergence of the electron probe (miliradians)."""

    sampling: float
    """Sampling of the model potential (Angstrom). Smaller values increase simulation times, but also increase the 
    maximum frequency which can be represented in reciprocal space via the equation ``|k| < min(Nx/(2*a), Ny/(2*b))`` 
    where ``Nx`` and ``Ny`` are the number of samples in the x and y directions (``a`` and ``b`` are the model 
    dimensions; see ``tiling`` below).  Choose a small enough value to ensure that you will be able to resolve 
    features of interest and that you will cover the virtual detector outer angle (usually around 0.1 A)."""

    tiling: int | tuple[int, int] | list[int] | list[tuple[int, int]]
    """Number of times to tile the projected cell; more tiling will give more realistic results at the cost of 
    simulation speed and memory usage. If a single integer is passed then tiling will be the same in both x and y 
    directions; if different tilings are desired in x and y (e.g. for non-square projected cells) then a tuple can be 
    passed instead. If multiple zone axes will be simulated (see ``zone`` below) then a list of either ints or tuples 
    can be passed and will be applied to each zone axis in the order they are specified."""

    zone: tuple[int, int, int] | list[tuple[int, int, int]]
    """Zone axes along which to project the structure and simulate images, in the form ``(h, k, l)``. If a 
    list is provided, a series of images will be simulated and saved for each zone axis in the list."""

    detectors: list[AnnularDetector | FlexibleAnnularDetector | SegmentedDetector | PixelatedDetector]
    """A list containing detector objects (representing, e.g., virtual HAADF / DF4 / etc detectors) to be used to 
    form images.  See the abTEM documentation for instructions on how to instantiate detector objects.  Note that the 
    largest outer radius of any detectors listed here must be less than the maximum outer radius set by sampling and 
    tiling (see sampling above for more info)."""

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
        assert self.device in ["gpu", "cpu"]
        if self.device == "gpu":
            assert self.gpu_number is not None, "Must set ``gpu_number`` if simulating on the GPU."
            assert self.gpu_number >= 0, "``gpu_number`` must be non-negative."

        for prm in [("max_batch", self.max_batch),
                    ("beam_energy", self.beam_energy),
                    ("convergence", self.convergence),
                    ("sampling", self.sampling),
                    ("thickness", self.thickness),
                    ("thickness_step", self.thickness_step),
                    ("slice_thickness", self.slice_thickness),
                    ("phonon_configs", self.phonon_configs),
                    ("tilt_mag", self.tilt_mag)
                    ]:
            # noinspection PyTypeChecker
            assert prm[1] >= 0, f"``{prm[0]}`` must be non-negative."

        assert self.thickness_step <= self.thickness, "``thickness_step`` must be <= ``total thickness``."
        assert self.slice_thickness <= self.thickness_step, "``slice_thickness`` muse be <= ``thickness_step``."

        if type(self.zone) is tuple:
            assert type(self.tiling) is int or type(self.tiling) is tuple, "``tiling`` cannot be a list if ``zone`` "\
                                                                           "is not."
        if type(self.zone) is list and type(self.tiling) is list:
            assert len(self.zone) == len(self.tiling), "Length mismatch between ``zone`` and ``tiling``."
        if type(self.tiling) is int:
            assert self.tiling > 0, "``tiling`` must be positive and non-zero."
        if type(self.tiling) is tuple:
            assert len(self.tiling) == 2, "``tiling`` must have two dimensions."
            for t in self.tiling:
                assert type(t) is int, "``tiling`` must contain only integers."
                assert t > 0, "All ``tiling`` dimesions must be positive and non-zero."
        if type(self.tiling) is list:
            for til in self.tiling:
                if type(til) is int:
                    assert til > 0, "All ``tiling``s must be positive and non-zero."
                if type(til) is tuple:
                    assert len(til) == 2, "``tiling`` must have two dimensions."
                    for t in til:
                        assert type(t) is int, "All ``tiling``s must contain only integers."
                        assert t > 0, "All ``tiling`` dimensions must be positive and non-zero."

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


# %% Select structure load method
# Note: ASE supports formats other than .cif; if you would like to use another format see their documentation at
#       https://wiki.fysik.dtu.dk/ase/ase/io/io.html
load_method: Literal[
    "fixed",        # Load via a fixed file path, given below, without a file picker window
    "interactive",  # Interactively select the file using a (native) file picker window
    "mp_api"        # Load a cif from Materials Project; requires an API key!
    ] = "mp_api"

fixed_path = Path("E:/Users/Charles/AlN.cif")  # Local path to .cif file
mp_id = "mp-2542"  # Materials Project ID for desired structure, including the leading "mp-"

match load_method:
    case "fixed":
        struct = io.read(fixed_path)
    case "interactive":
        interactive_path = Path(askopenfilename())
        struct = io.read(interactive_path)
    case "mp_api":
        from mp_api.client import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor
        from data_analysis_scripts.cse_secrets import MP_API_KEY
        mp_api_key = MP_API_KEY  # Personal API key (tied to MP account)

        with MPRester(mp_api_key) as mp:
            mp_struct = mp.get_structure_by_material_id(mp_id)
            print(mp_struct)
            struct = AseAtomsAdaptor().get_atoms(mp_struct)
    case _:
        raise NotImplementedError(f"Unrecognized file load method: {load_method}")

# %% Define the detector(s)
detector_list = []
save_dir = askdirectory()
haadf = AnnularDetector(inner=70,
                        outer=200,
                        save_file=save_dir)

# %% Define the parameters
# %% Define simulation parameters
# All parameters are required
prms = Parameters(
        device="gpu",
        gpu_number=0,
        max_batch=50,
        beam_energy=200E3,
        convergence=17.9,
        sampling=0.1,
        tiling=[(27, 30)],
        zone=[(1, 1, 0)],
        detectors=detector_list,
        thickness=200,
        thickness_step=20,
        slice_thickness=2,
        phonon_configs=0,
        phonon_sigmas={"Al": 0.0567,  # DOI: 10.1107/S0108767309004966
                       "Sc": 0.0567,  # No good source for this, assuming the same as Al
                       "N":  0.0593},
        seed=None,
        tilt_mag=0,
        tilt_angle=float(radians(0))  # Auto-convert degrees to radians
        )