import batman
from numpy import linspace, array, isnan
import matplotlib
from pathlib import Path
from pandas import read_csv
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


# This function return the coefficients for the
# quadratic parametrization of the limb darkening
def load_limb_coeff(limb_dark_path, profile):
    # Storing the file information into a dataframe
    df = read_csv(limb_dark_path, sep='\s+')
    # Filtering the dataframe to obtain only
    # the one with a quadratic profile
    row = df[df["profile"] == profile]
    # Mean of the coefficients
    mean_c1 = array(row["c1"], dtype=float).mean()
    mean_c2 = array(row["c2"], dtype=float).mean()
    mean_c3 = array(row["c3"], dtype=float).mean()
    mean_c4 = array(row["c4"], dtype=float).mean()
    return array([mean_c1, mean_c2, mean_c3, mean_c4])


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
    csv_profile_name = yaml_params["limb_dark"]
    # Correction in case of square root name
    limb_profile = "squareroot" if (
            csv_profile_name == "square-root"
    ) else csv_profile_name
    params.limb_dark = limb_profile
    # Coefficients limb darkening
    file_limb_dark = str(Path(
        input_folder_path, yaml_params["file_limb_dark"]
    ))
    limb_dark_coeff = load_limb_coeff(
        file_limb_dark, csv_profile_name
    )
    # limb darkening coefficients [c1, c2]
    params.u = limb_dark_coeff[~isnan(limb_dark_coeff)]
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
    plt.title(
        planet_name + " light curve with " + limb_profile +
        " limb darkening approximation"
    )
    plt.savefig(str(Path(
        output_folder_path,
        planet_name + "_" + limb_profile + "_assignment1_taskF.png"
    )))
    plt.show()
