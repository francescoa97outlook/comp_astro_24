import batman
from numpy import linspace, array, isnan
import matplotlib
from pathlib import Path
from pandas import read_csv
import matplotlib.pyplot as plt
from PyAstronomy import constants as c
matplotlib.use('Qt5Agg')


#

def load_limb_coeff(limb_dark_path, profile):
    """
    This function returns the coefficients for the parametrization
    of the limb darkening

    Args:
        limb_dark_path:
            string containing the path to the limb darkening
            csv file produced by https://exoctk.stsci.edu/limb_darkening
        profile:
            string containing the name of the profile. Possible values:
            ["quadratic", "exponential", "linear", logarithmic", "square-root"]

    Return:
        an array containing the mean values of each limb darkening coefficients
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
def transit_yaml(param_obj_list, output_folder_path, input_folder_path, multi):
    """
        This function calculates the transit light curves of the selected planets

        Args:
            param_obj_list:
                list of objects of Parameters class containing the keys and values
                of the chosen planets
            output_folder_path:
                string containing the output folder path where the results are saved
            input_folder_path:
                string containing the input folder path where the files needed for the
                algorithm are retrieved
            multi:
                flag to plot all planets light curves as single figures

        Return:

        """
    fig_single, ax_single = plt.subplots(1, 1, figsize=(10, 8))
    for yaml_params in param_obj_list:
        planet_name = yaml_params.get("planet_name")
        # object to store transit parameters
        params = batman.TransitParams()
        # time of inferior conjunction
        params.t0 = yaml_params.get("time0")
        # orbital period (in days)
        params.per = yaml_params.get("period")
        # planet radius (in units of stellar radii)
        planet_radius = yaml_params.get("planet_radius") * c.RJ
        stellar_radius = yaml_params.get("planet_radius") * c.RSun
        params.rp = planet_radius / stellar_radius
        # semi-major axis (in units of stellar radii)
        params.a = yaml_params.get("a")
        # orbital inclination (in degrees)
        params.inc = yaml_params.get("inc")
        # eccentricity
        params.ecc = yaml_params.get("ecc")
        # longitude of periastron (in degrees)
        params.w = yaml_params.get("w")
        # transit time between point 1 and 4 in hours
        t14h = yaml_params.get("t14h")
        # transit time in days
        t14d = t14h / 24
        # limb darkening model
        csv_profile_name = yaml_params.get("limb_dark")
        # Correction in case of square root name
        limb_profile = "squareroot" if (
                csv_profile_name == "square-root"
        ) else csv_profile_name
        params.limb_dark = limb_profile
        # Coefficients limb darkening
        file_limb_dark = str(Path(
            input_folder_path, yaml_params.get("file_limb_dark")
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
        t = linspace(-limit_transit + params.t0, limit_transit + params.t0, 5000)
        # initializes model
        m = batman.TransitModel(params, t)
        # calculates light curve
        flux = m.light_curve(params)
        # Plot the figure of the single planet
        if multi:
            fig_multi, ax_multi = plt.subplots(1, 1, figsize=(10, 8))
            ax_multi.plot(t, flux)
            ax_multi.set_xlabel("Time from central transit [days]")
            ax_multi.set_ylabel("Relative flux")
            ax_multi.set_title(
                planet_name + " light curve with " + limb_profile +
                " limb darkening approximation"
            )
            plt.savefig(str(Path(
                output_folder_path,
                planet_name + "_" + limb_profile + ".png"
            )))
            plt.close(fig_multi)
        #
        ax_single.plot(t, flux, label=planet_name + ", " + limb_profile)
    # Plot all the planets
    ax_single.set_xlabel("Time from central transit [days]")
    ax_single.set_ylabel("Relative flux")
    ax_single.set_title("Light curves")
    ax_single.legend(loc='lower right')
    plt.savefig(str(Path(
        output_folder_path, "planets_light_curves.png"
    )))
    plt.show()
    plt.close(fig_single)
