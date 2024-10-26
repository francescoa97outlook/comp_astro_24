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


def transit_yaml(yaml_params, output_folder_path, input_folder_path):
    planet_name = yaml_params["planet_name"]
    # object to store transit parameters
    params = batman.TransitParams()
    # time of inferior conjunction
    params.t0 = yaml_params["t0"]
    # orbital period (in days)
    params.per = yaml_params["per"]
    # planet radius (in units of stellar radii)
    params.rp = yaml_params["rp"]
    # semi-major axis (in units of stellar radii)
    params.a = yaml_params["a"]
    # orbital inclination (in degrees)
    params.inc = yaml_params["inc"]
    # eccentricity
    params.ecc = yaml_params["ecc"]
    # longitude of periastron (in degrees)
    params.w = yaml_params["w"]
    # transit time between point 1 and 4 in hours
    t14h = yaml_params["t14h"]
    # transit time in days
    t14d = t14h / 24
    # limb darkening model
    params.limb_dark = yaml_params["limb_dark"]
    # Coefficients limb darkening
    file_limb_dark = str(Path(input_folder_path, yaml_params["file_limb_dark"]))
    c1, e1, c2, e2 = load_limb_coeff(file_limb_dark)
    # limb darkening coefficients [c1, c2]
    params.u = [c1, c2]
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
    plt.title(planet_name)
    plt.savefig(str(Path(
        output_folder_path,
        planet_name + "_assignment1_taskF.png"
    )))
    plt.show()
