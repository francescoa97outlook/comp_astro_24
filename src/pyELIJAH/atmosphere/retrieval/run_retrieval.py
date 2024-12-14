from pathlib import Path
from multiprocessing import Process
from pyELIJAH import Parameters
from pyELIJAH.atmosphere.retrieval.Retrieval import Retrieval


def process_retrieval(i_ret, retrieval_yaml_file, input_folder, output_folder, parallel):
    """
    Execute the retrieval process for a specific YAML configuration.

    Args:
        i_ret:
            An integer identifier for the retrieval process.
        retrieval_yaml_file:
            A dictionary representing the YAML configuration file for the retrieval, including an "atmospheres" key.
        input_folder:
            A string containing the path to the folder where input YAML files are stored.
        output_folder:
            A string containing the path to the folder where output files will be saved.
        parallel:
            A boolean indicating whether the process is executed in parallel.

    Workflow:
        - Print the retrieval process identifier.
        - Iterate over the "atmospheres" key in the retrieval YAML file.
        - Load atmosphere parameters from the specified YAML files in the input folder.
        - For each atmosphere YAML file:
            - Iterate over the "planet" key in the atmosphere parameters.
            - Load planet parameters from the specified YAML files in the input folder.
            - Initialize a `Retrieval` object with all input parameters.
            - Perform the following actions with the `Retrieval` object:
                1. `calculate_radiative_model`: Compute the radiative model.
                2. `plot_observed_spectrum`: Generate a plot of the observed spectrum.
                3. `retrieval`: Execute the retrieval process.
    """
    print(f"Starting retrieval {i_ret}")
    for atmosphere_yaml_file in retrieval_yaml_file.get("atmospheres"):
        atmosphere_yaml = Parameters(Path(input_folder, atmosphere_yaml_file))
        for planet_yaml_file in atmosphere_yaml.get("planet"):
            planet_yaml = Parameters(Path(input_folder, planet_yaml_file))
            retrieval = Retrieval(
                input_folder, output_folder, retrieval_yaml_file, atmosphere_yaml, planet_yaml, i_ret, parallel
            )
            retrieval.calculate_radiative_model()
            retrieval.plot_observed_spectrum()
            retrieval.retrieval()


def run_retrieval(input_folder, output_folder, yaml_files, parallel):
    """
        Perform the retrieval process for multiple YAML configuration files.

        Args:
            input_folder:
                A string containing the path to the folder where input YAML files are stored.
            output_folder:
                A string containing the path to the folder where output files will be saved.
            yaml_files:
                A list of dictionaries, each representing a retrieval YAML configuration file.
            parallel:
                A boolean indicating whether to execute the retrieval processes in parallel.

        Workflow:
            - If parallel execution is enabled:
                - Create a list to store processes.
                - For each retrieval YAML file, initialize a new process targeting `process_retrieval` and start it.
                - Wait for all processes to complete using `join`.
            - If parallel execution is disabled:
                - Sequentially execute `process_retrieval` for each retrieval YAML file.
        """
    if parallel:
        list_process = list()
        for i_ret, retrieval_yaml_file in enumerate(yaml_files):
            task = (i_ret, retrieval_yaml_file, input_folder, output_folder, parallel)
            list_process.append(Process(target=process_retrieval, args=task))
            list_process[-1].start()
        for p in list_process:
            p.join()
    else:
        for i_ret, retrieval_yaml_file in enumerate(yaml_files):
            process_retrieval(i_ret, retrieval_yaml_file, input_folder, output_folder, parallel)
