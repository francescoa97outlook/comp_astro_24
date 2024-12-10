from pathlib import Path
import numpy as np
from corner import corner
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
            spectra = self.binned_spectrum.bin_model(
                self.atmosphere.calculate_model(
                    0, self.observed_spectrum.wavenumberGrid
                )
            )[1]
            #
            fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
            fig.suptitle(self.planet_name + " observed vs model - Iteration: " + str(index))
            axs[0].errorbar(
                self.observed_spectrum.wavelengthGrid,
                self.observed_spectrum.spectrum,
                self.observed_spectrum.errorBar, label='Observed Spectrum', color="r", alpha=0.8
            )
            axs[0].set_ylabel("Ratio planet-stellar radii")
            axs[0].legend()
            #
            axs[1].plot(
                self.observed_spectrum.wavelengthGrid,
                spectra,
                label=self.radiative_mod + " model", color="g", alpha=0.8
            )
            axs[1].set_ylabel("Ratio planet-stellar radii")
            axs[1].legend()
            #
            axs[2].plot(
                self.observed_spectrum.wavelengthGrid,
                spectra,
                label=self.radiative_mod + " model", color="g", alpha=0.5
            )
            axs[2].errorbar(
                self.observed_spectrum.wavelengthGrid,
                self.observed_spectrum.spectrum,
                self.observed_spectrum.errorBar, label='Observed Spectrum', color="r", alpha=0.5
            )
            axs[2].set_ylabel("Ratio planet-stellar radii")
            axs[2].legend()
            #
            axs[3].plot(
                self.observed_spectrum.wavelengthGrid,
                self.observed_spectrum.spectrum - spectra,
                label="Difference between observed and model", color="orange", alpha=0.8
            )
            axs[3].set_xlabel('Wavelength (um)')
            axs[3].set_ylabel("Ratio planet-stellar radii")
            axs[3].legend()
            plt.savefig(str(Path(
                self.output_folder, self.planet_name + "_" + str(index) + ".png"
            )))
            plt.show()
            plt.close(fig)
            #
            parameters = values[1][1]
            print("Retrieval results:\n", parameters)
            list_names = list()
            list_distributions = list()
            for key in parameters:
                list_names.append(key)
                list_distributions.append(parameters[key]["trace"])
            #
            fig = plt.figure(figsize=(12, 8))
            corner(
                data=np.array(list_distributions).T, labels=list_names, show_titles=True,
                title_fmt=".2f", title_kwargs={"fontsize": 12}, fig=fig
            )
            plt.savefig(str(Path(
                self.output_folder, self.planet_name + "_" + str(index) + "_corner_plot.png", color="orange"
            )))
            plt.show()
            plt.close(fig)
            #
