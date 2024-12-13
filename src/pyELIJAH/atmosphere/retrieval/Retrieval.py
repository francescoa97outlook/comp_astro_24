import pickle
from pathlib import Path
import numpy as np
from corner import corner
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmp
from taurex.data.spectrum import ObservedSpectrum
from taurex.optimizer.nestle import NestleOptimizer
from taurex.optimizer.multinest import MultiNestOptimizer
import mpi4py.MPI as MPI
import taurex.log
taurex.log.disableLogging()

from pyELIJAH.atmosphere.atmosphere.run_atmosphere import atmosphere_exc


class Retrieval:
    """
        A class for performing retrieval calculations for atmospheric models.
    """

    def __init__(self, input_folder, output_folder, retrieval_file, atmosphere_yaml_file, planet_yaml, i_ret):
        """
        Args:
            input_folder:
                A string representing the path to the folder containing input YAML files.
            output_folder:
                A string representing the path to the folder where output files will be saved.
            retrieval_file:
                A dictionary or YAML file containing the retrieval configuration.
            atmosphere_yaml_file:
                A dictionary or YAML file containing the atmospheric configuration.
            planet_yaml:
                A dictionary or YAML file containing the planet's configuration.
            i_ret:
                An integer representing the index of the retrieval, used for naming outputs.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.retrieval_file = retrieval_file
        self.atmosphere_yaml_file = atmosphere_yaml_file
        self.planet_yaml = planet_yaml
        self.planet_name = self.planet_yaml.get("planet_name")
        self.index_retrieval = i_ret
        if self.retrieval_file.get("output_retrieval_file") != "None":
            self.output_file = str(Path(
                self.output_folder, self.retrieval_file.get("output_retrieval_file"))
            )
        else:
            self.output_file = str(Path(self.output_folder, self.planet_name))
        self.output_file +=  f"_Retrieval_{int(self.index_retrieval)}"
        cmap = cmp["inferno"]
        self.colors = cmap(np.linspace(0, 1, 20))
        #
        self.atmosphere = None
        self.radiative_mod= None
        self.observed_spectrum = None
        self.binned_spectrum = None

    def calculate_radiative_model(self):
        """
        Calculates the radiative model for the retrieval process.

        Args:

        Workflow:
            - Executes the `atmosphere_exc` function to calculate the atmospheric model.
                - Inputs include the folder paths, atmosphere and planet
                configurations, and settings.
                - The `plot` argument is set to `False` to suppress visual
                outputs during this step.
            - Retrieves the type of radiative model to be used (`radiative_mod`)
            from the atmospheric YAML configuration.

        Returns:
            None
        """
        self.atmosphere = atmosphere_exc(self.input_folder, self.output_folder, self.atmosphere_yaml_file, self.planet_yaml, 0, False)
        self.radiative_mod = self.atmosphere_yaml_file.get("radiative_mod")
    
    def plot_observed_spectrum(self):
        """
        Plots the observed spectrum and compares it with the calculated
        radiative model.

        Args:

        Workflow:
            - Builds the atmospheric model using the specified radiative model
            type (`radiative_mod`).
            - Loads the observed spectrum from the file specified in the retrieval
            YAML configuration (`path_spectrum`).
            - Creates a binned version of the observed spectrum using `create_binner`.
            - Generates a plot:
                - The observed spectrum is displayed with error bars.
                - The calculated model spectrum is plotted for comparison.
            - Sets plot labels, titles, and legend.
            - Saves the plot as an image file named "<output_file>_raw_model.png".

        Returns:
            None
        """
        self.atmosphere.build_model(
            self.radiative_mod
        )
        self.observed_spectrum = ObservedSpectrum(str(Path(
            self.input_folder, self.retrieval_file.get("path_spectrum")[self.index_retrieval]
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
        plt.savefig(self.output_file + "_raw_model.png")
        plt.close(fig)

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def retrieval(self):
        """
        Performs the retrieval process to fit the observed spectrum
        to a radiative model.

        Args:

        Workflow:
            - Prompts the user to select parameters to fit from the available
            model parameters.
            - Configures the optimizer based on the retrieval configuration
            (`MultiNestOptimizer` or `NestleOptimizer`).
            - Sets up the forward model and the observed spectrum for fitting.
            - Enables fitting for the chosen parameters and sets their
            boundary conditions based on user input.
            - Runs the optimizer to find solutions and iteratively
            updates the model with optimized parameters.
            - For each iteration:
                - Compares the observed spectrum with the model spectrum.
                - Generates and saves plots for the observed spectrum,
                the model spectrum, and their differences.
                - Outputs the retrieval results in a corner plot and saves
                it as an image.
                - Saves the results, including observed data, model spectra,
                and optimized parameters, to a `.pkl` file.

        Returns:
            None
        """
        #
        #
        # Which parameters to fit and their prior boundaries:
        selected_params = list()
        selected_boundaries = list()
        parameters_to_fit = self.atmosphere_yaml_file.get("parameters_to_fit")
        for param in list(parameters_to_fit.keys()):
            selected_params.append(param)
            selected_boundaries.append(parameters_to_fit[param])
        # -------------------------------------------------------------------- #
        opt = NestleOptimizer(
            num_live_points=int(self.retrieval_file.get("num_live_points")),
        )
        # Forward model and observation:
        opt.set_model(self.atmosphere.model)
        opt.set_observed(self.observed_spectrum)
        for i in range(len(selected_params)):
            boundaries = selected_boundaries[i]
            opt.enable_fit(selected_params[i])
            opt.set_boundary(
                selected_params[i],
                [float(boundaries[0]), float(boundaries[1])]
            )
        # Lets loop and plot each solution!
        opt.fit()
        # -------------------------------------------------------------------- #
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
            fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 12), constrained_layout=True, dpi=500)
            fig.suptitle(self.planet_name + " observed vs model - Iteration: " + str(index))
            axs[0].errorbar(
                self.observed_spectrum.wavelengthGrid,
                self.observed_spectrum.spectrum,
                self.observed_spectrum.errorBar, label='Observed Spectrum', color=self.colors[-5], alpha=0.7
            )
            axs[0].set_ylabel("Ratio planet-stellar radii")
            axs[0].legend()
            #
            axs[1].plot(
                self.observed_spectrum.wavelengthGrid,
                spectra,
                label=self.radiative_mod + " model", color=self.colors[4], alpha=0.7
            )
            axs[1].set_ylabel("Ratio planet-stellar radii")
            axs[1].legend()
            #
            axs[2].plot(
                self.observed_spectrum.wavelengthGrid,
                spectra,
                label=self.radiative_mod + " model", color=self.colors[4], alpha=0.3
            )
            axs[2].errorbar(
                self.observed_spectrum.wavelengthGrid,
                self.observed_spectrum.spectrum,
                self.observed_spectrum.errorBar, label='Observed Spectrum', color=self.colors[-5], alpha=0.3
            )
            axs[2].set_ylabel("Ratio planet-stellar radii")
            axs[2].legend()
            #
            axs[3].plot(
                self.observed_spectrum.wavelengthGrid,
                self.observed_spectrum.spectrum - spectra,
                label="Difference between observed and model", color=self.colors[10], alpha=0.7
            )
            axs[3].set_xlabel('Wavelength (um)')
            axs[3].set_ylabel("Ratio planet-stellar radii")
            axs[3].legend()
            #
            string_output = self.output_file + f"_Iteration_{index}"
            plt.savefig(string_output + "_results_comparison.png")
            # plt.show()
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
            fig = plt.figure(figsize=(20, 20), dpi=500)
            corner(
                data=np.array(list_distributions).T, labels=list_names, show_titles=True,
                title_fmt=".2f", fig=fig, color=self.colors[4]
            )
            plt.savefig(str(Path(
                string_output + "_corner_plot.png"
            )))
            # plt.show()
            plt.close(fig)
            #
            with open(string_output + "_results.pkl", "wb") as f:
                pickle.dump(self.observed_spectrum.wavelengthGrid, f)
                pickle.dump(self.observed_spectrum.spectrum, f)
                pickle.dump(self.observed_spectrum.errorBar, f)
                pickle.dump(spectra, f)
                pickle.dump(solution, f)
                pickle.dump(optimized_map, f)
                pickle.dump(optimized_value, f)
                pickle.dump(values, f)


    # # noinspection PyTypeChecker,PyUnresolvedReferences
    # def retrieval(self):
    #     """
    #     Performs the retrieval process to fit the observed spectrum
    #     to a radiative model.
    #
    #     Args:
    #
    #     Workflow:
    #         - Prompts the user to select parameters to fit from the available
    #         model parameters.
    #         - Configures the optimizer based on the retrieval configuration
    #         (`MultiNestOptimizer` or `NestleOptimizer`).
    #         - Sets up the forward model and the observed spectrum for fitting.
    #         - Enables fitting for the chosen parameters and sets their
    #         boundary conditions based on user input.
    #         - Runs the optimizer to find solutions and iteratively
    #         updates the model with optimized parameters.
    #         - For each iteration:
    #             - Compares the observed spectrum with the model spectrum.
    #             - Generates and saves plots for the observed spectrum,
    #             the model spectrum, and their differences.
    #             - Outputs the retrieval results in a corner plot and saves
    #             it as an image.
    #             - Saves the results, including observed data, model spectra,
    #             and optimized parameters, to a `.pkl` file.
    #
    #     Returns:
    #         None
    #     """
    #     #
    #     rank = None
    #     selected_params = None
    #     selected_boundaries = None
    #     do_mpi = int(self.retrieval_file.get("mpi"))
    #     if do_mpi:
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
    #     rank_0_or_no_mpi = rank == 0 or not do_mpi
    #     #
    #     if rank_0_or_no_mpi:
    #         # Which parameters to fit and their prior boundaries:
    #         selected_params = list()
    #         selected_boundaries = list()
    #         parameters_to_fit = self.atmosphere_yaml_file.get("parameters_to_fit")
    #         for param in list(parameters_to_fit.keys()):
    #             selected_params.append(param)
    #             selected_boundaries.append(parameters_to_fit[param])
    #     # -------------------------------------------------------------------- #
    #     if do_mpi:
    #         opt = MultiNestOptimizer(
    #             multi_nest_path=self.output_folder,
    #             num_live_points=int(self.retrieval_file.get("num_live_points")),
    #         )
    #     else:
    #         opt = NestleOptimizer(
    #             num_live_points=int(self.retrieval_file.get("num_live_points")),
    #         )
    #     # Forward model and observation:
    #     opt.set_model(self.atmosphere.model)
    #     opt.set_observed(self.observed_spectrum)
    #     for i in range(len(selected_params)):
    #         boundaries = selected_boundaries[i]
    #         opt.enable_fit(selected_params[i])
    #         opt.set_boundary(
    #             selected_params[i],
    #             [float(boundaries[0]), float(boundaries[1])]
    #         )
    #     #
    #     # Lets loop and plot each solution!
    #     opt.fit()
    #     # -------------------------------------------------------------------- #
    #     if rank_0_or_no_mpi:
    #         index = -1
    #         for solution, optimized_map, optimized_value, values in opt.get_solution():
    #             index += 1
    #             opt.update_model(optimized_map)
    #             spectra = self.binned_spectrum.bin_model(
    #                 self.atmosphere.calculate_model(
    #                     0, self.observed_spectrum.wavenumberGrid
    #                 )
    #             )[1]
    #             #
    #             fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 12), constrained_layout=True, dpi=500)
    #             fig.suptitle(self.planet_name + " observed vs model - Iteration: " + str(index))
    #             axs[0].errorbar(
    #                 self.observed_spectrum.wavelengthGrid,
    #                 self.observed_spectrum.spectrum,
    #                 self.observed_spectrum.errorBar, label='Observed Spectrum', color=self.colors[-5], alpha=0.7
    #             )
    #             axs[0].set_ylabel("Ratio planet-stellar radii")
    #             axs[0].legend()
    #             #
    #             axs[1].plot(
    #                 self.observed_spectrum.wavelengthGrid,
    #                 spectra,
    #                 label=self.radiative_mod + " model", color=self.colors[4], alpha=0.7
    #             )
    #             axs[1].set_ylabel("Ratio planet-stellar radii")
    #             axs[1].legend()
    #             #
    #             axs[2].plot(
    #                 self.observed_spectrum.wavelengthGrid,
    #                 spectra,
    #                 label=self.radiative_mod + " model", color=self.colors[4], alpha=0.3
    #             )
    #             axs[2].errorbar(
    #                 self.observed_spectrum.wavelengthGrid,
    #                 self.observed_spectrum.spectrum,
    #                 self.observed_spectrum.errorBar, label='Observed Spectrum', color=self.colors[-5], alpha=0.3
    #             )
    #             axs[2].set_ylabel("Ratio planet-stellar radii")
    #             axs[2].legend()
    #             #
    #             axs[3].plot(
    #                 self.observed_spectrum.wavelengthGrid,
    #                 self.observed_spectrum.spectrum - spectra,
    #                 label="Difference between observed and model", color=self.colors[10], alpha=0.7
    #             )
    #             axs[3].set_xlabel('Wavelength (um)')
    #             axs[3].set_ylabel("Ratio planet-stellar radii")
    #             axs[3].legend()
    #             #
    #             string_output = self.output_file + f"_Iteration_{index}"
    #             plt.savefig(string_output + "_results_comparison.png")
    #             # plt.show()
    #             plt.close(fig)
    #             #
    #             parameters = values[1][1]
    #             print("Retrieval results:\n", parameters)
    #             list_names = list()
    #             list_distributions = list()
    #             for key in parameters:
    #                 list_names.append(key)
    #                 list_distributions.append(parameters[key]["trace"])
    #             #
    #             fig = plt.figure(figsize=(20,20), dpi=500)
    #             corner(
    #                 data=np.array(list_distributions).T, labels=list_names, show_titles=True,
    #                 title_fmt=".2f", fig=fig, color=self.colors[4]
    #             )
    #             plt.savefig(str(Path(
    #                 string_output + "_corner_plot.png"
    #             )))
    #             # plt.show()
    #             plt.close(fig)
    #             #
    #             with open(string_output + "_results.pkl", "wb") as f:
    #                 pickle.dump(self.observed_spectrum.wavelengthGrid, f)
    #                 pickle.dump(self.observed_spectrum.spectrum, f)
    #                 pickle.dump(self.observed_spectrum.errorBar, f)
    #                 pickle.dump(spectra, f)
    #                 pickle.dump(solution, f)
    #                 pickle.dump(optimized_map, f)
    #                 pickle.dump(optimized_value, f)
    #                 pickle.dump(values, f)