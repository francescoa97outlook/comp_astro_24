import batman
from numpy import linspace
import matplotlib
from pathlib import Path
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set a variables containing the absolute path of the starting folder
    path_default = str(Path(__file__).parent.resolve()).replace(
        str(Path("src", "daneel", "detection")), ""
    )
    # planet chosen to complete this task
    # All the following parameters values where retrieved from the link:
    # https://exoplanetarchive.ipac.caltech.edu/overview/WASP-107%20b#planet_WASP-107-b_collapsible
    # The period and Rp/R✶ parameters were selected from Kokori
    # while the others from Anderson
    planet_name = "WASP107-b"
    # object to store transit parameters
    params = batman.TransitParams()
    # time of inferior conjunction
    params.t0 = 0.
    # orbital period (in days)
    params.per = 5.72148926
    # planet radius (in units of stellar radii)
    params.rp = 0.1446
    # semi-major axis (in units of stellar radii)
    params.a = 18.2
    # orbital inclination (in degrees)
    params.inc = 89.7
    # eccentricity
    params.ecc = 0.
    # longitude of periastron (in degrees)
    params.w = 90.
    # limb darkening model
    params.limb_dark = "nonlinear"
    # limb darkening coefficients [u1, u2, u3, u4]
    params.u = [0.5, 0.1, 0.1, -0.1]
    # transit time between point 1 and 4 in hours
    t14h = 2.753
    # transit time in days
    t14d = t14h / 24
    limit_transit = t14d / 2 + 0.3 * t14d
    # times at which to calculate light curve (days)
    t = linspace(-limit_transit, limit_transit, 5000)
    # initializes model
    m = batman.TransitModel(params, t)
    # calculates light curve
    flux = m.light_curve(params)
    # Plot the figure
    plt.plot(t, flux)
    plt.xlabel("Time from central transit")
    plt.ylabel("Relative flux")
    plt.savefig(str(Path(path_default, "src", "daneel", "detection", "WASP107-b_assignment1_taskF.png")))
    plt.savefig(str(Path(path_default, "Results", "assignment1", "WASP107-b_assignment1_taskF.png")))
    plt.show()
