from pyELIJAH.atmosphere.Retrieval import Retrieval


def run_atmosphere(input_folder, output_folder, yaml_files):
    for atmosphere_yaml_file in yaml_files:
        retrieval = Retrieval(input_folder, output_folder, atmosphere_yaml_file)
        retrieval.calculate_radiative_model()
        retrieval.plot_observed_spectrum()
        retrieval.retrieval()
