from pathlib import Path

from pyELIJAH import Parameters
from pyELIJAH.atmosphere.atmosphere.Atmosphere import Atmosphere


def run_atmosphere(input_folder, output_folder, yaml_files):
    for atmosphere_yaml_file in yaml_files:
        for planet in atmosphere_yaml_file.get("planet"):
            planet_yaml = Parameters(Path(input_folder, planet))
            atmosphere_exc(input_folder, output_folder, atmosphere_yaml_file, planet_yaml)


def atmosphere_exc(input_folder, output_folder, atmosphere_yaml_file, planet_yaml, plot=True):
    atmosphere = Atmosphere(input_folder, output_folder, atmosphere_yaml_file, planet_yaml, plot)
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
