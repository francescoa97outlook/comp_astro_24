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
    """
        A class for generating and analyzing atmospheric models based on input
        configurations for planets and their atmospheres. It uses Taurex module.
    """

    def __init__(self, input_folder, output_folder, atmosphere_yaml_file, planet_yaml, i_atmo, plot=True):
        """
            Initializes the Atmosphere class to process and analyze atmospheric models for planets.

            Args:
                input_folder:
                    A string representing the path to the folder containing input YAML files.
                output_folder:
                    A string representing the path to the folder where output files will be saved.
                atmosphere_yaml_file:
                    A dictionary containing the YAML configuration for the atmosphere.
                    This includes information such as the output file name for the atmosphere
                    and model-specific parameters.
                planet_yaml:
                    An instance of the `Parameters` class containing the planet's
                    configuration and associated properties.
                i_atmo:
                    An integer representing the index of the atmosphere being processed,
                    used for naming outputs.
                plot:
                    A boolean flag indicating whether to generate plots during processing.
                    Default is True.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.atmosphere_yaml_file = atmosphere_yaml_file
        self.planet_yaml = planet_yaml
        self.index_atmosphere = i_atmo
        if self.atmosphere_yaml_file.get("output_atmosphere_file") != "None":
            self.output_file = str(Path(self.output_folder, self.atmosphere_yaml_file.get("output_atmosphere_file")))
        else:
            self.output_file = str(Path(self.output_folder, self.planet_name))
        self.output_file += f"_Atmosphere_{int(self.index_atmosphere)}"
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
        """
        Generates the necessary profiles for the atmosphere, planet,
        and star based on the YAML configuration files.

        Args:


        Workflow:
            - Checks the "temperature_profile" key in `atmosphere_yaml_file`:
                - If "isothermal", initializes an `Isothermal` temperature profile
                  using the planet's equilibrium temperature.
                - Otherwise, initializes a `Guillot2010` temperature profile.
            - Assigns the planet's name from the YAML configuration.
            - Initializes the planet's properties (`Planet`) using radius
              and mass from `planet_yaml`.
            - Initializes the star's properties (`BlackbodyStar`) using temperature
              and radius from `planet_yaml`.

        Returns:
            None

        """
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

    # noinspection PyUnresolvedReferences
    def read_opacities_and_create_chemistry(self):
        """
        Reads opacity data, sets up chemistry models, and generates
        cross-section plots if required.

        Args:


        Workflow:
            - Clears any cached opacity data and sets the paths for cross-section
              (`xsec_path`) and CIA (`hitran_path`) files.
            - Initializes a free chemistry model (`TaurexChemistry`)
              with fill gases and a mixing ratio.
            - Iterates over the molecules specified in `dict_molecs`:
                - Computes the mixing ratio for each molecule based on the configuration.
                - Adds the molecule as a constant gas to the chemistry model.
                - If plotting is enabled, logs the mixing ratio and plots the
                cross-section for each molecule.
            - Handles exceptions for missing molecules in HITRAN and logs a
              message if plotting is enabled.
            - If `plot` is enabled:
                - Creates and saves cross-section plots for each molecule,
                  showing wave number vs cross-section for specific temperature
                  and pressure conditions.

        Returns:
            None
        """

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
            if self.plot:
                print(f"{molec} mixing ratio: {mix_ratio}")
            try:
                temp_press_mix.append(temp)
                self.cross_sections.append(OpacityCache()[molec])
            except Exception as _:
                if self.plot:
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
                #
                plt.savefig(self.output_file + f"_{molecs[i]}_cross_sections.png")
                plt.close(fig)
        #

    def create_binning_obj(self):
        """
        Creates a binning object and initializes the wavenumber grid.

        Args:

        Workflow:
            - Generates a sorted wavenumber grid (`wngrid`) by converting from
              logarithmic wavelength spacing.
            - Initializes a `SimpleBinner` object using the generated `wngrid`.

        Returns:
            None
        """

        self.wngrid = np.sort(10000 / np.logspace(-0.4, 1.1, 1000))
        self.binner = SimpleBinner(wngrid=self.wngrid)
    
    def build_model(self, type_model):
        """
        Builds the atmospheric model based on the specified model type.

        Args:
            type_model:
                A string specifying the type of model to build. Options are:
                - "Transmission": Builds a `TransmissionModel`.
                - "Emission": Builds an `EmissionModel`.
                - Any other value defaults to a `DirectImageModel`.

        Workflow:
            - Initializes the model object based on the provided `type_model` using the planet,
              temperature profile, chemistry, star, and atmospheric
              pressure range from `atmosphere_yaml_file`.
            - Adds the following contributions to the model:
                - Absorption contributions (`AbsorptionContribution`).
                - CIA contributions (`CIAContribution`) based on `cia_pairs`
                  from the YAML configuration.
                - Rayleigh scattering contributions (`RayleighContribution`).
            - Calls the `build` method on the model to set up all the profiles and
              prepare the model for further calculations.

        Returns:
            None
        """

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
        """
        Calculates the atmospheric model, with optional binning of the results.

        Args:
            binning:
                A boolean indicating whether to apply binning to the calculated model.
                Default is False.
            wavenumberGrid:
                An optional array specifying the wavenumber grid to use for the model calculation.
                If None, the default grid is used.

        Workflow:
            - If `binning` is True, bins the model output using the `SimpleBinner` object.
            - If `binning` is False, calculates the model directly on the specified or
              default wavenumber grid.

        Returns:
            The calculated atmospheric model, either binned or unbinned,
            depending on the `binning` argument.
        """
        if binning:
            return self.binner.bin_model(self.model.model(wavenumberGrid))
        else:
            return self.model.model(wavenumberGrid)

    def plot_gases(self):
        """
        Plots the mixing ratios of active and inactive gases in the atmosphere.

            Args:

            Workflow:
                - Creates a plot showing the mixing ratios of active and
                 inactive gases against the pressure profile.
                - Uses logarithmic scaling for both axes.
                - Inverts the y-axis to represent pressure decreasing with altitude.
                - Saves the plot as an image file named "<output_file>_gases.png".

            Returns:
                None
        """
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
            plt.savefig(self.output_file + "_gases.png")
            plt.show()
            plt.close(fig)

    def plot_flux(self, result_model):
        """
        Plots the calculated flux spectrum and saves it as an image and data file.

            Args:
                result_model:
                    A tuple containing:
                        - `wn`: Wavenumber grid.
                        - `ratio_rp_rs`: Ratio of planet-to-stellar radii.
                        - `tau`: Optical depth (unused in plotting).
                        - Additional model output (unused in plotting).

            Workflow:
                - Plots the ratio of planet-to-stellar radii against the wavelength.
                - Saves the plot as an image file named "<output_file>_flux.png".
                - Saves the flux data to a `.dat` file with columns: wavelength,
                radius ratio, and an estimated uncertainty.

            Returns:
                None
        """
        if self.plot:
            fig, ax = plt.subplots(1, 1, figsize = (12, 8))
            wn, ratio_rp_rs, tau, _ = result_model
            ax.plot(10000 / wn, ratio_rp_rs)
            ax.set_xlabel("Wavelengths (um)")
            ax.set_ylabel("Ratio planet-stellar radii")
            ax.set_title(self.planet_name + " flux")
            plt.savefig(self.output_file + "_flux.png")
            plt.show()
            plt.close(fig)
            # File path
            # Write to file
            ratio_rp_rs_pow = np.zeros(len(ratio_rp_rs)) + np.std(ratio_rp_rs)
            data = np.column_stack((10000 / wn, ratio_rp_rs, ratio_rp_rs_pow))
            # Save to a .dat file without an empty first row
            np.savetxt(self.output_file + "_spectrum.dat", data, fmt="%.15e", comments="")


    def compare_models(self, binning=False):
        """
        Compares the calculated atmospheric models for Transmission, Emission,
        and Direct Imaging.

        Args:
            binning:
                A boolean indicating whether to apply binning to the calculated models.
                Default is False.

        Workflow:
            - Builds and calculates the atmospheric model for "Transmission".
            - Builds and calculates the atmospheric model for "Emission".
            - Builds and calculates the atmospheric model for "Direct Image".
            - Plots the ratio of planet-to-stellar radii for each model against the wavelength.
            - Creates a subplot for each model:
                - The first subplot displays the Transmission model.
                - The second subplot displays the Emission model.
                - The third subplot displays the Direct Image model.
            - Saves the comparison plot as an image file named "<output_file>_compare.png".

        Returns:
            None
        """
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
            ax[0].set_title('Transmission')
            ax[0].set_xlabel('Wavelength (um)')
            ax[0].set_ylabel("Ratio planet-stellar radii")
            ax[1].set_title('Emission')
            ax[1].set_xlabel('Wavelength (um)')
            ax[1].set_ylabel("Ratio planet-stellar radii")
            ax[2].set_title('Direct Image')
            ax[2].set_xlabel('Wavelength (um)')
            ax[2].set_ylabel("Ratio planet-stellar radii")
            plt.savefig(self.output_file + "_compare.png")
            plt.close(fig)

