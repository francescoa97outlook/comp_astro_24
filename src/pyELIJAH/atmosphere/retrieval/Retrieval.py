from pathlib import Path
from matplotlib import pyplot as plt
from taurex.data.spectrum import ObservedSpectrum
from taurex.optimizer.nestle import NestleOptimizer
import taurex.log
taurex.log.disableLogging()

from pyELIJAH.atmosphere.atmosphere.run_atmosphere import atmosphere_exc


class Retrieval:

    def __init__(self, input_folder, output_folder, retrieval_file, atmosphere_yaml_file, planet_yaml):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.retrieval_file = retrieval_file
        self.atmosphere_yaml_file = atmosphere_yaml_file
        self.planet_yaml = planet_yaml
        self.planet_name = self.planet_yaml.get("planet_name")
        #
        self.atmosphere = None
        self.radiative_mod= None
        self.observed_spectrum = None
        self.binned_spectrum = None

    def calculate_radiative_model(self):
        self.atmosphere = atmosphere_exc(self.input_folder, self.output_folder, self.atmosphere_yaml_file, self.planet_yaml, False)
        self.radiative_mod = self.atmosphere_yaml_file.get("radiative_mod")
    
    def plot_observed_spectrum(self):
        self.atmosphere.build_model(
            self.radiative_mod
        )
        self.observed_spectrum = ObservedSpectrum(str(Path(
            self.input_folder, self.retrieval_file.get("path_spectrum")
        )))
        self.binned_spectrum = self.observed_spectrum.create_binner()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.errorbar(
            self.observed_spectrum.wavelengthGrid,
            self.observed_spectrum.spectrum,
            self.observed_spectrum.errorBar,
            label='Observed Spectrum'
        )
        ax.plot(
            self.observed_spectrum.wavelengthGrid,
            self.binned_spectrum.bin_model(
                self.atmosphere.calculate_model(
                    0, self.observed_spectrum.wavenumberGrid
                ))[1],
            label=self.radiative_mod + " model"
        )
        ax.legend()
        ax.set_xlabel("Wavelengths (um)")
        ax.set_ylabel("Pressure")
        ax.set_title(self.planet_name + " observed vs model - Raw")
        ax.legend()
        plt.savefig(str(Path(self.output_folder, self.planet_name + "_raw_model.png")))
        plt.close(fig)

    def retrieval(self):
        list_fit_param = list(self.atmosphere.model.fittingParameters.keys())
        print(
            "\n\nParameters available for the retrieval:",
            ', '.join(f"{i}: {param}" for i, param in enumerate(list_fit_param))
        )
        print("Which parameters do you want to fit? Write '0,2,5,..' based on the indices")
        chosen_parameters = input().split(',')
        opt = NestleOptimizer(
            num_live_points=int(self.retrieval_file.get("num_live_points"))
        )
        # Forward model and observation:
        opt.set_model(self.atmosphere.model)
        opt.set_observed(self.observed_spectrum)
        # Which parameters to fit and their prior boundaries:
        for index in chosen_parameters:
            param = list_fit_param[int(index)]
            value = self.atmosphere.model[param]
            print(f"Boundaries for {param}? i.e. {value - 0.5 * value},{value + 0.5 * value}")
            boundaries = input().split(',')
            opt.enable_fit(param)
            opt.set_boundary(param, [float(boundaries[0]), float(boundaries[1])])
        #
        # Lets loop and plot each solution!
        opt.fit()
        index = -1
        for solution, optimized_map, optimized_value, values in opt.get_solution():
            index += 1
            opt.update_model(optimized_map)
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.errorbar(
                self.observed_spectrum.wavelengthGrid,
                self.observed_spectrum.spectrum,
                self.observed_spectrum.errorBar, label='Observed Spectrum'
            )
            #
            ax.plot(
                self.observed_spectrum.wavelengthGrid,
                self.binned_spectrum.bin_model(
                    self.atmosphere.calculate_model(
                        0, self.observed_spectrum.wavenumberGrid
                    ))[1],
                label=self.radiative_mod + " model"
            )
            ax.legend()
            ax.set_xscale('log')
            ax.set_title(self.planet_name + " observed vs model - Iteration: " + str(index))
            ax.set_xlabel('Wavelength (um)')
            ax.set_ylabel("Ratio planet-stellar radii")
            plt.savefig(str(Path(
                self.output_folder, self.planet_name + "_" + str(index) + ".png"
            )))
            plt.show()
            plt.close(fig)