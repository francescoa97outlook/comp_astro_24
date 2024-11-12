import batman
from numpy import linspace, array, isnan
import matplotlib
from pathlib import Path
from pandas import read_csv
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


#

def load_limb_coeff(limb_dark_path, profile):
    """
    This function return the coefficients for the parametrization
    of the limb darkening

    Args:
        limb_dark_path:
            string containing the path to the limb darkening
            csv file produced by https://exoctk.stsci.edu/limb_darkening
        profile:
            string containing the name of the profile. Possible values:
            ["quadratic", "exponential", "linear", logarithmic", "square-root"]

    Return:
        an array containing the mean of at maximum 4 limb darkening coefficients
    """
    # Storing the file information into a dataframe
    df = read_csv(limb_dark_path, sep='\s+')
    # Filtering the dataframe to obtain only
    # the one with the selected profile
    row = df[df["profile"] == profile]
    # Mean of the coefficients
    mean_c1 = array(row["c1"], dtype=float).mean()
    mean_c2 = array(row["c2"], dtype=float).mean()
    mean_c3 = array(row["c3"], dtype=float).mean()
    mean_c4 = array(row["c4"], dtype=float).mean()
    return array([mean_c1, mean_c2, mean_c3, mean_c4])


# using batman package
def transit_yaml(yaml_params, output_folder_path, input_folder_path):
    """
        This function calculates the light curve of the transit

        Args:
            yaml_params:
                object of Parameters class containing the keys and values
                of the stored in the yaml file
            output_folder_path:
                string containing the path to the output folder, where to save results
            input_folder_path:
                string containing the path to the input folder, where to retrieve
                the files needed for the algorithm

        Return:
            an array containing the mean of at maximum 4 limb darkening coefficients
        """
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
    # limb darkening coefficients
    # In this way they could be from 1 to 4
    # depending on the method
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
