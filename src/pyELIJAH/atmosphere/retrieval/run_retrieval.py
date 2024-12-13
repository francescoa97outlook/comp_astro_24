from pathlib import Path

from pyELIJAH import Parameters
from pyELIJAH.atmosphere.retrieval.Retrieval import Retrieval


def run_retrieval(input_folder, output_folder, yaml_files):
    """
    This function performs the retrieval process for a set of YAML configuration files.

    Args:
        input_folder:
            A string containing the path to the folder where input YAML files are stored.
        output_folder:
            A string containing the path to the folder where output files will be saved.
        yaml_files:
            A list of dictionaries, each representing a retrieval YAML configuration file. Each dictionary
            should include an "atmospheres" key containing a list of atmosphere YAML file paths.

    Workflow:
        - For each retrieval YAML file:
            - Load atmosphere YAML files specified under the "atmospheres" key.
            - For each atmosphere YAML file:
                - Load planet YAML files specified under the "planet" key.
                - Create a `Retrieval` object using the input and output folders, the retrieval YAML file,
                  the atmosphere parameters, and the planet parameters.
                - Perform the following actions with the `Retrieval` object:
                    1. `calculate_radiative_model`: Calculate the radiative model.
                    2. `plot_observed_spectrum`: Plot the observed spectrum.
                    3. `retrieval`: Run the retrieval process.
    """
    for i_ret, retrieval_yaml_file in enumerate(yaml_files):
        for atmosphere_yaml_file in retrieval_yaml_file.get("atmospheres"):
            atmosphere_yaml = Parameters(Path(input_folder, atmosphere_yaml_file))
            for planet_yaml_file in atmosphere_yaml.get("planet"):
                planet_yaml = Parameters(Path(input_folder, planet_yaml_file))
                retrieval = Retrieval(
                    input_folder, output_folder, retrieval_yaml_file, atmosphere_yaml, planet_yaml, i_ret
                )
                retrieval.calculate_radiative_model()
                retrieval.plot_observed_spectrum()
                retrieval.retrieval()
