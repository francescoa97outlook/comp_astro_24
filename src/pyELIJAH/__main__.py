import datetime
import argparse
from pathlib import Path

from pyELIJAH.parameters.parameters import Parameters
from pyELIJAH.detection.transit.transit_yaml import transit_yaml


# Function that is called at the beginning
# or directly from pyELIJAH command
def main():
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
        help="Input par file to pass",
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
    # Define an expected input folder with command -limbd.
    # It must be a string (str) and is not required.
    # It will be used in the code with the key directory_limb
    parser.add_argument(
        "-limbd",
        "--directory_limb",
        dest="directory_limb",
        type=str,
        required=False,
        help="""
        Directory of the limb darkening file (or others). 
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
        help="Atmospheric Characterisazion from "
             "input transmission spectrum",
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
    # the planets parameters
    input_params = Parameters(args.input_file).params
    # This is defined to establish the input folder
    # If the -limbd param is set, the path will be the one
    # defined from the argument.
    # If not, it will be the default path
    if args.directory_limb:
        input_folder_path_light_curve = str(Path(
            args.directory_limb
        ))
    else:
        input_folder_path_light_curve = str(Path(
            path_default.replace(
                str(Path("src", "pyELIJAH")), ""
            ),
            "Data", "assignment1",
        ))
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
            "Results", "assignment1",
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
            input_params, output_folder_path_light_curve,
            input_folder_path_light_curve
        )
    # End of the program
    finish = datetime.datetime.now()
    print(f"pyELIJAH finishes at {finish}")


# Main function that will be executed by
# launching normally the script
if __name__ == "__main__":
    main()
