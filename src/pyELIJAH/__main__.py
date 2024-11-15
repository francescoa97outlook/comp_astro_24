import datetime
import argparse
from pathlib import Path

import numpy as np

from pyELIJAH.parameters.parameters import Parameters
from pyELIJAH.detection.transit.transit_yaml import transit_yaml


def main():
    """
    Function that is called at the beginning
    or directly from pyELIJAH command.
    If this function is called by pyELIJAH command,
    the user can choose between different argument to
    be passed.
    - pyelijah -h, --help: shows the help message
    - pyelijah -i INPUT_FILE, --input INPUT_FILE:
      input list file to pass as argument (a txt file containing the list of yaml
      files, each one in a row)
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

    - pyelijah -d, --detect: initialise detection algorithms for Exoplanets
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
        help="input list file to pass as argument (a txt file containing the list of planets, "
             "each one in a row)",
    )
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
    #
    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        required=False,
        help="Initialise detection algorithms for Exoplanets",
        action="store_true",
    )
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
    # Define an expected command -single.
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
    # This command will parse (convert) the arguments
    # with the respective expected commands.
    args = parser.parse_args()
    # Set a variables containing the absolute
    # path of the starting folder
    path_default = str(Path(__file__).parent)
    # Launch pyELIJAH
    start = datetime.datetime.now()
    print(f"pyELIJAH starts at {start}")
    # Define the required input folder in which are retrieved
    # the planets files
    input_list_files = args.input_file
    list_files = np.genfromtxt(input_list_files, dtype=str)
    # This is defined to establish the input folder
    # If the -id param is set, the path will be the one
    # defined from the argument.
    # If not, it will be the default path
    if args.directory_input:
        input_folder_path_light_curve = str(Path(
            args.directory_input
        ))
    else:
        input_folder_path_light_curve = str(Path(
            path_default.replace(
                str(Path("src", "pyELIJAH")), ""
            ),
            "Data",
        ))
    # Retrieve multi arguments
    multi = args.multi_plot
    #
    param_obj_list = list()
    for file_planet in list_files:
        param_obj_list.append(
            Parameters(Path(input_folder_path_light_curve, file_planet)).params
        )
    # This is defined to establish the output folder
    # If the -res param is set, the path will be the one
    # defined from the argument.
    # If not, it will be the default path
    if args.directory_results:
        output_folder_path_light_curve = str(Path(
            args.directory_results
        ))
    else:
        output_folder_path_light_curve = str(Path(
            path_default.replace(
                str(Path("src", "pyELIJAH")), ""
            ),
            "Results",
        ))
    #
    if args.detect:
        pass
    #
    if args.atmosphere:
        pass
    # If the -t command is passed as argument, the code will
    # calculate the transit light curve using
    # the transit_yaml function
    if args.transit:
        # call the transit.py file
        transit_yaml(
            param_obj_list, output_folder_path_light_curve,
            input_folder_path_light_curve, multi
        )
    # End of the program
    finish = datetime.datetime.now()
    print(f"pyELIJAH finishes at {finish}")


"""
    Main function that will be executed by
    launching normally the script
"""
if __name__ == "__main__":
    main()
