import abtem
from collections.abc import Collection, Sequence
from numpy import cos, sin
from packaging import version
from warnings import warn


def simulate_stem(potential: abtem.Potential,
                  detectors: abtem.detectors.BaseDetector | Collection[abtem.detectors.BaseDetector],
                  convergence: float,
                  beam_energy: float = 200E3,
                  scan: abtem.scan.BaseScan | None = None,
                  tilt_mag: float = 0,
                  tilt_angle: float = 0,
                  aberrations: dict[str, float] | abtem.transfer.Aberrations | None = None,
                  eager: bool = True,
                  prism: bool = False,
                  prism_interp_factor: int = 1,
                  **kwargs)\
        -> abtem.Images | abtem.measurements.BaseMeasurements | abtem.array.ComputableList | tuple:
    """Simple wrapper function for the normal abTEM multislice STEM simulation workflow.

    Args:
        potential: An abTEM Potential object to be scanned.
        detectors: One or a collection of several abTEM detector objects (typically AnnularDetector,
            FlexibleAnnularDetector, SegmentedDetector, or PixelatedDetector).
        convergence: The semiangle of convergence of the simulated electron probe (milliradians).
        beam_energy: Optional, the energy of the electron wavefunction (eV).  If None, defaults to 200E3.
        scan: Optional, an abTEM scan object defining the scan to perform. If None, defaults to scanning
            the full extent of the potential.
        tilt_mag: Optional, the magnitude of tilting (milliradians) of the electron probe (equivalent to tilting
            the sample _for small tilts only_). Defaults to zero.
        tilt_angle: Optional, the angle (radians anticlockwise with respect to +x) of tilting of the electron
            probe. Defaults to zero.
        aberrations: Optional, a dictionary of aberration symbols and their corresponding values.  For details see
            https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html. Also
            accepts an abTEM ``Aberrations`` object. Defaults to no aberrations.
        eager: Optional, whether to return the lazy abTEM object (False) or perform the multislice computations
            and return the computed result (True). Defaults to True.
        prism: Optional, whether to use the PRISM algorithm _without_ interpolation (True) or the traditional
            multislice algorithm (False) for the simulation. Defaults to False.
        prism_interp_factor: Optional, the interpolation factor for use with the PRISM algorithm.
            Defaults to 1 (no interpolation).
        kwargs: Optional, keyword arguments passed into the simulation function (either multislice or PRISM).
    Returns:
        The results of the simulation.
    """
    if not version.parse(abtem.__version__) >= version.parse("1.0.0"):
        raise RuntimeError("This method will only work with abTEM versions >= 1.0.0 due to breaking changes in abTEM. "
                           "Please update you abTEM version.")

    print("Simulating...")
    if scan is None:
        scan = abtem.scan.GridScan(potential=potential)
    tilt = (tilt_mag * sin(tilt_angle), tilt_mag * cos(tilt_angle))

    if prism:
        if tilt != (0, 0):
            warn("Non-zero tilt specified: tilts are not supported under the PRISM algorithm. Set ``prism=False`` or "
                 "rotate your model potential.")
        probe = abtem.SMatrix(semiangle_cutoff=convergence,
                              energy=beam_energy,
                              potential=potential,
                              interpolation=prism_interp_factor,
                              store_on_host=True)
        ctf = abtem.CTF(semiangle_cutoff=convergence,
                        aberration_coefficients=aberrations,
                        energy=beam_energy)
        measurement = probe.scan(scan=scan,
                                 detectors=detectors,
                                 ctf=ctf,
                                 **kwargs)
    else:
        probe = abtem.Probe(semiangle_cutoff=convergence,
                            energy=beam_energy,
                            sampling=potential.sampling,
                            tilt=tilt,
                            aberrations=aberrations)
        probe.match_grid(potential)
        measurement = probe.scan(potential=potential,
                                 scan=scan,
                                 detectors=detectors,
                                 **kwargs)
    if eager:
        measurement.compute()
    return measurement


def preview_stem_result(result: abtem.Images | abtem.measurements.BaseMeasurements | abtem.array.ComputableList,
                        titles: Sequence | None = None)\
        -> None:
    """Quickly preview STEM simulation results to verify that a simulation went as expected.

    This function intentially has no options (to keep it simple), and you will likely want to implement your own
    display functions which are more specifically relevant to your experiment.

    Args:
        result: Images or measurements resulting from a STEM simulation.
        titles: Optional, a collection containing a title to print on each plot. If None (default), don't add titles.
    """
    if not isinstance(result, abtem.array.ComputableList):
        result = [result]  # Wrap into single element list for the following loop

    if titles is not None:
        if len(result) != len(titles):
            raise RuntimeError("If titles are specified, one must be specified for each image.")
    else:
        titles = [None]*len(result)

    for res, title in zip(result, titles):
        match res:
            case abtem.Images():  # Simple annular detectors
                img = res
            case abtem.measurements.PolarMeasurements():  # Flexible and segmented annular detectors
                # noinspection PyUnresolvedReferences
                img = res.integrate()  # Full annular and azimuthal range
            case abtem.measurements.DiffractionPatterns():  # Pixelated (4D) detectors
                # noinspection PyUnresolvedReferences
                img = res.integrated_center_of_mass()
            case _:
                warn("Fell through all guards; check ``type(result)``")
                img = None

        if img is not None:
            # Hardcoded 4x FFT interpolation (works very well in most cases)
            img.interpolate(sampling=(img.sampling[0] / 4, img.sampling[1] / 4)).show(title=title)


def _test_potential() -> abtem.Potential:
    """Returns an example potential (and pops up a view of it) in order to test that simulations are working."""
    from ase.build import bcc100
    from ase.visualize import view
    structure = bcc100("Fe", size=(6, 6, 20), orthogonal=True, periodic=True)
    structure.symbols[700] = "Au"
    view(structure)
    return abtem.Potential(structure,
                           sampling=0.04,
                           projection="infinite",
                           parametrization="kirkland",
                           slice_thickness=2)


if __name__ == "__main__":
    # Run this as a script to test that the importable functions are working
    # Simulating this small test potential should take < 1 minute on reasonable hardware
    from abtem.detectors import (FlexibleAnnularDetector,
                                 AnnularDetector,
                                 SegmentedDetector,
                                 PixelatedDetector)

    # When calling ``simulate_stem`` from another script, ensure you set up your abTEM configuration
    # This can either be done like this (in the script) or in a configuration file to use the same settings
    #   for all scripts in a given python environment
    abtem.config.set({"device":              "gpu",  # Configure abTEM to run on the GPU
                      "dask.lazy":           True,  # Setting to False can be useful for debugging
                      "dask.chunk-size":     "128 MB",  # Standard L3 cache size (per core)
                      "dask.chunk-size-gpu": "2048 MB"})  # Remember to leave space for overhead

    # noinspection PyTypeChecker
    result = simulate_stem(potential=_test_potential(),
                           detectors=[AnnularDetector(70, 200),
                                      FlexibleAnnularDetector(),
                                      SegmentedDetector(1, 4,
                                                        9, 36, rotation=3.98),
                                      PixelatedDetector(200, resample="uniform",
                                                        reciprocal_space=True)],
                           convergence=20)

    # Should generate and show 4 images of the test potential if everything went well
    preview_stem_result(result, ["HAADF", "Flex (HABF)", "DF4 (Sum)", "iCOM"])

#%%
