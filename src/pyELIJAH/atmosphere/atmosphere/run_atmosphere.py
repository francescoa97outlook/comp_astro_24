from pathlib import Path

from pyELIJAH import Parameters
from pyELIJAH.atmosphere.atmosphere.Atmosphere import Atmosphere


def run_atmosphere(input_folder, output_folder, yaml_files):
    """
        This function processes atmosphere YAML configuration files and performs
        atmospheric calculations for specified planets.

        Args:
            input_folder:
                A string containing the path to the folder where input YAML files are stored.
            output_folder:
                A string containing the path to the folder where output files will be saved.
            yaml_files:
                A list of dictionaries, each representing an atmosphere YAML configuration file.
                Each dictionary  should include a "planet" key containing a
                list of planet YAML file paths.

        Workflow:
            - For each atmosphere YAML file:
                - Retrieve the planet YAML files specified under the "planet" key.
                - Execute atmospheric calculations for each planet using the `atmosphere_exc` function.

        """
    for i, atmosphere_yaml_file in enumerate(yaml_files):
        for planet in atmosphere_yaml_file.get("planet"):
            planet_yaml = Parameters(Path(input_folder, planet))
            atmosphere_exc(input_folder, output_folder, atmosphere_yaml_file, planet_yaml, i)


def atmosphere_exc(input_folder, output_folder, atmosphere_yaml_file, planet_yaml, index_atmo=1, plot=True):
    """
        This function executes the atmospheric model generation and analysis for a specific atmosphere and planet.

        Args:
            input_folder:
                A string containing the path to the folder where input YAML files are stored.
            output_folder:
                A string containing the path to the folder where output files will be saved.
            atmosphere_yaml_file:
                A dictionary representing the YAML configuration for the atmosphere.
            planet_yaml:
                An instance of the `Parameters` class containing the planet configuration.
            index_atmo:
                An integer indicating the index of the atmosphere. Default is 1.
            plot:
                A boolean indicating whether to generate plots for the atmospheric
                calculations. Default is True.

        Workflow:
            - Initialize the `Atmosphere` object with the provided configurations.
            - Generate atmospheric profiles.
            - Read opacities and set up the chemical models.
            - Create binning objects for radiative transfer calculations.
            - Build the radiative transfer model.
            - Calculate the atmospheric model using the specified binning resolution.
            - Generate plots for gases, fluxes, and model comparisons.

        Return:
            An `Atmosphere` object containing the results of the atmospheric calculations.

        """
    atmosphere = Atmosphere(input_folder, output_folder, atmosphere_yaml_file, planet_yaml, index_atmo, plot)
    atmosphere.generate_profiles()
    atmosphere.read_opacities_and_create_chemistry()
    atmosphere.create_binning_obj()
    atmosphere.build_model(
        atmosphere_yaml_file.get("radiative_mod")
    )
    result_model = atmosphere.calculate_model(int(atmosphere_yaml_file.get("binning")))
    atmosphere.plot_gases()
    atmosphere.plot_flux(result_model)
    atmosphere.compare_models(int(atmosphere_yaml_file.get("binning")))
    return atmosphere
