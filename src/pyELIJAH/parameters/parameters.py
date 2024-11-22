import os
import yaml


class Parameters:
    """
        This class is used to store the parameters
        inside a yaml file in a dictionary
    """
    def __init__(self, input_file):
        """
        This init the parameter class by reading the yaml file

        Args:
            input_file: path of the yaml file
        """
        if os.path.exists(input_file) and os.path.isfile(input_file):
            with open(input_file) as in_f:
                self.params = yaml.load(in_f, Loader=yaml.FullLoader)

        for par in list(self.params.keys()):
            if self.params[par] == "None":
                self.params[par] = None

    def get(self, param):
        """
        Args:
            param: key of the parameter to retrieve

        Returns:
            Value of the parameter key
        """
        if param in self.params.keys():
            return self.params[param]
        else:
            return None
