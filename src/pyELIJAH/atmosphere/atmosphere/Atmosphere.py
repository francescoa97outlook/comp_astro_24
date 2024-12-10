from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import taurex.log
taurex.log.disableLogging()
from taurex.cache import OpacityCache, CIACache
from taurex.temperature import Guillot2010
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry
from taurex.chemistry import ConstantGas
from taurex.model import TransmissionModel
from taurex.contributions import AbsorptionContribution
from taurex.contributions import CIAContribution
from taurex.contributions import RayleighContribution
from taurex.model import EmissionModel, DirectImageModel
from taurex.binning import SimpleBinner
from taurex.temperature import Isothermal


class Atmosphere:
    def __init__(self, input_folder, output_folder, atmosphere_yaml_file, planet_yaml, plot=True):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.atmosphere_yaml_file = atmosphere_yaml_file
        self.planet_yaml = planet_yaml
        self.plot = plot
        self.planet_name = None
        self.planet = None
        self.star = None
        self.chemistry = None
        self.temperature_profile = None
        self.model = None
        self.cross_sections = None
        self.binner = None
        self.wngrid = None

    def generate_profiles(self):
        # PROFILES
        if self.atmosphere_yaml_file.get("temperature_profile") == "isothermal":
            self.temperature_profile = Isothermal(T=float(self.planet_yaml.get("planet_temperature_eq")))
        else:
            self.temperature_profile = Guillot2010(T_irr=float(self.planet_yaml.get("planet_temperature_eq")))
        # Guillot profile
        self.planet_name = self.planet_yaml.get("planet_name")
        # Planets
        self.planet = Planet(
            planet_radius=float(self.planet_yaml.get("planet_radius")),
            planet_mass=float(self.planet_yaml.get("planet_mass"))
        )
        # Planet star
        self.star = BlackbodyStar(
            temperature=float(self.planet_yaml.get("stellar_temperature")),
            radius=float(self.planet_yaml.get("stellar_radius"))
        )

    def read_opacities_and_create_chemistry(self):
        # Now lets point the xsection and cia cachers to our files:
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(str(Path(self.input_folder, self.atmosphere_yaml_file.get("xsec_path"))))
        CIACache().set_cia_path(str(Path(self.input_folder, self.atmosphere_yaml_file.get("hitran_path"))))
        dict_molecs = self.atmosphere_yaml_file.get("dict_molecs")
        self.cross_sections = []
        # Free chemistry model
        self.chemistry = TaurexChemistry(fill_gases=self.atmosphere_yaml_file.get("fill_gases"), ratio=0.172)
        temp_press_mix = []
        molecs = list(dict_molecs.keys())
        for i, molec in enumerate(molecs):
            temp = dict_molecs[molec]
            if int(self.atmosphere_yaml_file.get("random_molecule")):
                mix_ratio = np.random.uniform(10**float(temp[2]), 10**float(temp[3]))
            else:
                mix_ratio = 10**float(temp[2])
            self.chemistry.addGas(ConstantGas(
                molec, mix_ratio=float(mix_ratio)
            ))
            print(f"{molec} mixing ratio: {mix_ratio}")
            try:
                temp_press_mix.append(temp)
                self.cross_sections.append(OpacityCache()[molec])
            except Exception as _:
                print(molec + " not found in hitran")
                continue
        #
        if self.plot:
            for i, cross_sect in enumerate(self.cross_sections):
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.set_xlabel("Wave number")
                ax.set_ylabel("Cross section")
                ax.set_title(self.planet_name + " cross sections " + molecs[i])
                ax.plot(
                    10000 / cross_sect.wavenumberGrid,
                    cross_sect.opacity(temp_press_mix[i][0], 10**temp_press_mix[i][1]),
                    label=f"{molecs[i]}, T={temp_press_mix[i][0]} K, P={10 ** temp_press_mix[i][1]} Pa",
                    alpha=0.5
                )
                ax.legend()
                plt.savefig(str(Path(
                    self.output_folder, self.planet_name + "_" + molecs[i] + "_cross_sections.png"
                )))
                plt.close(fig)
        #

    def create_binning_obj(self):
        self.wngrid = np.sort(10000 / np.logspace(-0.4, 1.1, 1000))
        self.binner = SimpleBinner(wngrid=self.wngrid)
    
    def build_model(self, type_model):
        # BUILDING THE MODEL
        if type_model == "Transmission":
            self.model = TransmissionModel(
                planet=self.planet,
                temperature_profile=self.temperature_profile,
                chemistry=self.chemistry,
                star=self.star,
                atm_min_pressure=float(self.atmosphere_yaml_file.get("atm_min_pressure")),
                atm_max_pressure=float(self.atmosphere_yaml_file.get("atm_max_pressure")),
                nlayers=int(self.atmosphere_yaml_file.get("nlayers"))
            )
        elif type_model == "Emission":
            self.model = EmissionModel(
                planet=self.planet,
                temperature_profile=self.temperature_profile,
                chemistry=self.chemistry,
                star=self.star,
                atm_min_pressure=float(self.atmosphere_yaml_file.get("atm_min_pressure")),
                atm_max_pressure=float(self.atmosphere_yaml_file.get("atm_max_pressure")),
                nlayers=int(self.atmosphere_yaml_file.get("nlayers"))
            )
        else:
            self.model = DirectImageModel(
                planet=self.planet,
                temperature_profile=self.temperature_profile,
                chemistry=self.chemistry,
                star=self.star,
                atm_min_pressure=float(self.atmosphere_yaml_file.get("atm_min_pressure")),
                atm_max_pressure=float(self.atmosphere_yaml_file.get("atm_max_pressure")),
                nlayers=int(self.atmosphere_yaml_file.get("nlayers"))
            )
        # Add absorption contribution
        self.model.add_contribution(AbsorptionContribution())
        # And some CIA for good measure:
        self.model.add_contribution(
            CIAContribution(cia_pairs=self.atmosphere_yaml_file.get("cia_pairs"))
        )
        # And rayleigh
        self.model.add_contribution(RayleighContribution())
        # Build it to setup all the profiles
        self.model.build()

    def calculate_model(self, binning=False, wavenumberGrid=None):
        if binning:
            return self.binner.bin_model(self.model.model(wavenumberGrid))
        else:
            return self.model.model(wavenumberGrid)

    def plot_gases(self):
        if self.plot:
            fig, ax = plt.subplots(1, 1, figsize = (12, 8))
            #
            for x, gasname in enumerate(self.model.chemistry.activeGases):
                ax.plot(
                    self.model.chemistry.activeGasMixProfile[x],
                    self.model.pressureProfile / 1e5, label=gasname
                )
            for x, gasname in enumerate(self.model.chemistry.inactiveGases):
                ax.plot(
                    self.model.chemistry.inactiveGasMixProfile[x],
                    self.model.pressureProfile / 1e5, label=gasname
                )
            ax.invert_yaxis()
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("Mix in ratio")
            ax.set_ylabel("Pressure (Pa)")
            ax.set_title(self.planet_name + " atmosphere chemistry")
            ax.legend()
            plt.savefig(str(Path(self.output_folder, self.planet_name + "_gases.png")))
            plt.show()
            plt.close(fig)

    def plot_flux(self, result_model):
        if self.plot:
            if self.planet_yaml.get("output_atmosphere_file") != "None":
                plot_name = str(Path(self.output_folder, self.planet_yaml.get("output_atmosphere_file")))
            else:
                plot_name = str(Path(self.output_folder, self.planet_name + "_flux.png"))
            fig, ax = plt.subplots(1, 1, figsize = (12, 8))
            wn, ratio_rp_rs, tau, _ = result_model
            ax.plot(10000 / wn, ratio_rp_rs)
            ax.set_xlabel("Wavelengths (um)")
            ax.set_ylabel("Ratio planet-stellar radii")
            ax.set_title(self.planet_name + " flux")
            plt.savefig(plot_name)
            plt.show()
            plt.close(fig)
            # File path
            output_file = str(Path(self.output_folder, self.planet_name + "_spectrum.dat"))
            # Write to file
            ratio_rp_rs_pow = np.zeros(len(ratio_rp_rs)) + np.std(ratio_rp_rs)
            data = np.column_stack((10000 / wn, ratio_rp_rs, ratio_rp_rs_pow))
            # Save to a .dat file without an empty first row
            np.savetxt(output_file, data, fmt="%.15e", comments="")


    def compare_models(self, binning=False):
        if self.plot:
            self.build_model("Transmission")
            tm_model = self.calculate_model(binning)
            self.build_model("Emission")
            em_model = self.calculate_model(binning)
            self.build_model("Direct Image")
            dim_model = self.calculate_model(binning)
            #
            fig, ax = plt.subplots(3, 1, figsize = (12, 8), constrained_layout=True)
            wn, ratio_rp_rs, _, _ = tm_model
            ax[0].plot(10000 / wn, ratio_rp_rs)
            #
            wn, ratio_rp_rs, _, _ = em_model
            ax[1].plot(10000 / wn, ratio_rp_rs)
            #
            wn, ratio_rp_rs, _, _ = dim_model
            ax[2].plot(10000 / wn, ratio_rp_rs)
            #
            ax[0].set_xscale('log')
            ax[0].set_title('Transmission')
            ax[0].set_xlabel('Wavelength (um)')
            ax[0].set_ylabel("Ratio planet-stellar radii")
            ax[1].set_xscale('log')
            ax[1].set_title('Emission')
            ax[1].set_xlabel('Wavelength (um)')
            ax[1].set_ylabel("Ratio planet-stellar radii")
            ax[2].set_xscale('log')
            ax[2].set_title('Direct Image')
            ax[2].set_xlabel('Wavelength (um)')
            ax[2].set_ylabel("Ratio planet-stellar radii")
            plt.savefig(str(Path(self.output_folder, self.planet_name + "_compare.png")))
            plt.close(fig)

