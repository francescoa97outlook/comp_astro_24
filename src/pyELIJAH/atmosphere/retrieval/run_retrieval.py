from pathlib import Path

from pyELIJAH import Parameters
from pyELIJAH.atmosphere.retrieval.Retrieval import Retrieval


def run_retrieval(input_folder, output_folder, yaml_files):
    for retrieval_yaml_file in yaml_files:
        for atmosphere_yaml_file in retrieval_yaml_file.get("atmospheres"):
            atmosphere_yaml = Parameters(Path(input_folder, atmosphere_yaml_file))
            for planet_yaml_file in atmosphere_yaml.get("planet"):
                planet_yaml = Parameters(Path(input_folder, planet_yaml_file))
                retrieval = Retrieval(input_folder, output_folder, retrieval_yaml_file, atmosphere_yaml, planet_yaml)
                retrieval.calculate_radiative_model()
                retrieval.plot_observed_spectrum()
                retrieval.retrieval()
