import datetime
import argparse
from pathlib import Path

from pyELIJAH.parameters.parameters import Parameters
from pyELIJAH.detection.transit.transit_yaml import transit_yaml
from pyELIJAH.detection.machine_learning.machine_learning_yaml import machine_learning
from pyELIJAH.dream.gan_model import gan_model


def main():
    """
    Function that is called at the beginning
    or directly from pyELIJAH command.
    If this function is called by pyELIJAH command,
    the user can choose between different argument to
    be passed.
    - pyelijah -h, --help: shows the help message
    - pyelijah -i INPUT_FILE, --input INPUT_FILE:
      a yaml file containing three list, one for each main modules. Each list
      contains yaml_files used in the corresponding module.
    - pyelijah -t, --transit: select the transit method
    - pyelijah -id DIRECTORY_INPUT, --directory_input DIRECTORY_INPUT:
      Directory of the input files (yaml, limb darkening...).
      If the user wants to set another folder to get their input data, use this option.
      In this case, just put the absolute path of the folder. Otherwise, put
      their data inside the Data folder (default one) and do not use this option.
    - pyelijah -res DIRECTORY_RESULTS, --directory_results DIRECTORY_RESULTS:
      directory where to store the results.
      If the user wants to set another folder to store their output data, use this option.
      In this case, just put the absolute path of the folder. Otherwise, the output
      will be stored in the Results folder.
    - pyelijah -multi, --multi_plot: flag to plot all planets information singularly
    - pyelijah -d, --detect: Command that is used to accept a limited set of strings
      concerning the machine learning algorithm used to detect
      exoplanets in a set of data.
      The accepted values are: "svm" for support vector machine,
      "nn" for neural network, "cnn" for convolutional neural network.
    - pyelijah -dr, --dream: used to call the GAN architecture that allows
      to 'dream' a new exoplanetary transit light curve
    - pyelijah -a, --atmosphere: atmospheric characterisation from input transmission spectrum
    Args:

    Returns:

    """
    # Create a ArgumentParser object to work
    # with script argument
    parser = argparse.ArgumentParser()
    # Define an expected input file with command -i.
    # It must be a string (str) and is required.
    # It will be used in the code with the key input_file
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="a yaml file containing three list, "
             "one for each main modules. Each list"
             "contains yaml_files used in the corresponding module.",
    )
    # ---------------------------------- #
    # Define an expected command -t.
    # It is not required and will be used with key transit
    # to launch the script transit_yaml.py
    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        required=False,
        help="Select the transit method",
        action="store_true",
    )
    # ---------------------------------- #
    # Define an expected input folder with command -id.
    # It must be a string (str) and is not required.
    # It will be used in the code with the key directory_input
    parser.add_argument(
        "-id",
        "--input_directory",
        dest="directory_input",
        type=str,
        required=False,
        help="""
        Directory of the input files (yaml, limb darkening...). 
        If you want to set another folder to get your 
        input data, use this option. In this case,  
        just put the absolute path of the folder.  
        Otherwise, put your data inside the Data 
        folder (default one) and do not use this option.
        """
    )
    # ---------------------------------- #
    # Define an expected output folder with command -res.
    # It must be a string (str) and is not required.
    # It will be used in the code with the key directory_results
    parser.add_argument(
        "-res",
        "--directory_results",
        dest="directory_results",
        type=str,
        required=False,
        help="""
            Directory where to store the results. 
            If you want to set another folder to store your 
            output data, use this option. In this case,  
            just put the absolute path of the folder. Otherwise, 
            the output will be stored in the Results folder.
            """
    )
    # ---------------------------------- #
    # Command that is used to accept a limited set of strings
    # concerning the machine learning algorithm used to detect
    # exoplanets in a set of data
    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        required=False,
        type=str,
        help="""
        Command that is used to accept a limited set of strings 
        concerning the machine learning algorithm used to detect 
        exoplanets in a set of data.
        The accepted values are: "svm" for support vector machine, 
        "nn" for neural network, "cnn" for convolutional neural network.
        """,
    )
    # ---------------------------------- #
    #
    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        required=False,
        help="Atmospheric Characterisation from "
             "input transmission spectrum",
        action="store_true",
    )
    # ---------------------------------- #
    # Define an expected command -multi.
    # It is not required and will be used with key multi_plot
    # to plot all planets information singularly
    parser.add_argument(
        "-multi",
        "--multi_plot",
        dest="multi_plot",
        required=False,
        help="Flag to plot all planets information singularly",
        action="store_true",
    )
    # ---------------------------------- #
    # Define an expected command -multi.
    # It is not required and will be used to
    # "create" a new exoplanetary transit light curve
    parser.add_argument(
        "-dr"
        "--dream",
        dest="dream",
        required=False,
        help="command used to call the GAN architecture that"
             "allows to 'dream' a new exoplanetary transit light curve",
        action="store_true"
    )
    # ------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------- #
    # This command will parse (convert) the arguments
    # with the respective expected commands.
    args = parser.parse_args()
    # Set a variables containing the absolute
    # path of the starting folder
    path_default = str(Path(__file__).parent)
    # Launch pyELIJAH
    start = datetime.datetime.now()
    print(f"pyELIJAH starts at {start}")
    # ------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------- #
    # This is defined to establish the input folder
    # If the -id param is set, the path will be the one
    # defined from the argument.
    # If not, it will be the default path
    if args.directory_input:
        input_folder = str(Path(
            args.directory_input
        ))
    else:
        input_folder = str(Path(
            path_default.replace(
                str(Path("..", "pyELIJAH")), ""
            ),
            "../../Data",
        ))
    # ---------------------------------- #
    # This is defined to establish the output folder
    # If the -res param is set, the path will be the one
    # defined from the argument.
    # If not, it will be the default path
    if args.directory_results:
        output_folder = str(Path(
            args.directory_results
        ))
    else:
        output_folder = str(Path(
            path_default.replace(
                str(Path("..", "pyELIJAH")), ""
            ),
            "../../Results",
        ))
    # ---------------------------------- #
    # Define the required input folder in which are retrieved
    # the planets files
    input_file = args.input_file
    # If the input file is a txt it means it contains
    # the list of planets for the transit/atmosphere functions
    # Otherwise, is a yaml file for the machine learning part
    if not input_file.endswith(".yaml"):
        print(f"Input file {input_file} is not a .yaml file.")
        return
    params_input_file = Parameters(Path(input_file))
    files_yaml_transit = params_input_file.get("list_for_transit")
    files_yaml_ml = params_input_file.get("list_for_ML")
    files_yaml_atmosphere = params_input_file.get("list_for_atmosphere")
    # ---------------------------------- #
    # Retrieve multi arguments
    multi = args.multi_plot
    # ---------------------------------- #
    # If the -t command is passed as argument, the code will
    # calculate the transit light curve using
    # the transit_yaml function
    if args.transit:
        if len(files_yaml_transit) > 0:
            params_transit_list = list()
            for file in files_yaml_transit:
                params_transit_list.append(Parameters(Path(input_folder, file)))
            # call the transit_yaml.py file
            transit_yaml(
                params_transit_list, output_folder, input_folder, multi
            )
        else:
            print("Error in command list. Check the arguments")
    # ---------------------------------- #
    # Detect machine learning
    if args.detect:
        if len(files_yaml_ml) > 0:
            model = args.detect
            params_ml_list = list()
            for file in files_yaml_ml:
                params_ml_list.append(Parameters(Path(input_folder, file)))
                machine_learning(input_folder, output_folder, model, params_ml_list)
        else:
            print("Error in command list. Check the arguments")
    # ---------------------------------- #
    # Dream machine learning - GAN
    if args.dream:
        if len(files_yaml_ml) > 0:
            params_ml_list = list()
            for file in files_yaml_ml:
                params_ml_list.append(Parameters(Path(input_folder, file)))
                gan_model(input_folder, output_folder, params_ml_list)
        else:
            print("Error in command list. Check the arguments")
    # ---------------------------------- #
    #
    if args.atmosphere:
        if len(files_yaml_atmosphere) > 0:
            print("TODO")
        else:
            print("Error in command list. Check the arguments")
        pass
    # ------------------------------------------------------------------------------- #
    # End of the program
    finish = datetime.datetime.now()
    print(f"pyELIJAH finishes at {finish}")


"""
    Main function that will be executed by
    launching normally the script
"""
if __name__ == "__main__":
    main()
