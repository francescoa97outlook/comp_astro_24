import batman
from numpy import linspace, array
import matplotlib
from pathlib import Path
from pandas import read_csv
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


# This function return the coefficients for the
# quadratic parametrization of the limb darkening
def load_limb_coeff(limb_dark_path):
    # Storing the file information into a dataframe
    df = read_csv(limb_dark_path, sep='\s+')
    # Filtering the dataframe to obtain only
    # the one with a quadratic profile
    row = df[df["profile"] == "quadratic"]
    # Mean of the coefficients
    mean_c1 = array(row["c1"], dtype=float).mean()
    mean_e1 = array(row["e1"], dtype=float).mean()
    mean_c2 = array(row["c2"], dtype=float).mean()
    mean_e2 = array(row["e2"], dtype=float).mean()
    return mean_c1, mean_e1, mean_c2, mean_e2


if __name__ == "__main__":
    # Set a variables containing the absolute
    # path of the starting folder
    path_default = (
        str(Path(__file__).parent.resolve()).replace(
            str(Path("src", "pyELIJAH", "detection", "transit")), ""
        )
    )
    # planet chosen to complete this task
    # All the following parameter values where
    # retrieved from the link:
    # https://exoplanetarchive.ipac.caltech.edu/overview/WASP-107%20b#planet_WASP-107-b_collapsible
    # The period and Rp/R_star parameters were selected from Kokori
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
    params.limb_dark = "quadratic"
    # Coefficients limb darkening
    file_limb_dark = str(Path(
        path_default, "Data", "assignment1",
        "limb_dark_wasp107b.txt"
    ))
    c1, e1, c2, e2 = load_limb_coeff(file_limb_dark)
    # limb darkening coefficients [c1, c2]
    params.u = [c1, c2]
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
    plt.title(planet_name + " light curve")
    plt.savefig(str(Path(
        path_default, "Results", "assignment1",
        "WASP107-b_assignment1_taskF.png"
    )))
    plt.show()
