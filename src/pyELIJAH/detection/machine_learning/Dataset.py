from pathlib import Path
from pandas import DataFrame, read_csv
from sklearn.utils import shuffle
from numpy import array

from pyELIJAH.detection.machine_learning.LightFluxProcessor import LightFluxProcessor


def np_X_Y_from_df(df, Y_valid):
    """
    Shuffle and process of data

    Args:
        df: pandas DataFrame containing the data
        Y_valid: boolean value indicating whether the labels are inside the dataframe
    """
    # Shuffle the data
    df = shuffle(df)
    # X process
    df_X = df.drop(["LABEL"], axis=1)
    X = array(df_X)
    # Y process if labels are provided
    if Y_valid:
        Y_raw = array(df["LABEL"]).reshape((len(df["LABEL"]), 1))
        Y = array(Y_raw == 2)
        Y = Y.ravel()
    else:
        Y = None
    return X, Y


class DatasetTrainDev:
    """
    Class to read and process the data
    """

    def __init__(self, input_path, model_ml=None,
                 train_file="exo_train.csv", dev_file="exo_dev.csv",
                 array_lenght=3136, image_size=56):
        """
        Init of the datasets

        Args:
            input_path (str): Path to the input files
            model_ml (str): Machine Learning model decided. Used only for the cnn model 
            train_file (str): Name of the csv file containing the training data
            dev_file (str): Name of the csv file containing the development data
            array_lenght (int): Lenght of the flux array after the slice. Used only for the cnn model
            image_size (int): Size of the square image in which the flux arry is reshaped. Used only for the cnn model
        """
        #
        print("Loading datasets...")
        #
        path_train = str(Path(input_path, train_file))
        path_dev = str(Path(input_path, dev_file))
        #
        df_train = read_csv(path_train, encoding="ISO-8859-1")
        df_dev = read_csv(path_dev, encoding="ISO-8859-1")
        #
        # Generate X and Y dataframe sets
        df_train_x = df_train.drop("LABEL", axis=1)
        df_train_y = df_train.LABEL
        # Define if label column is present or not. If present
        # it will be considered
        if "LABEL" in df_dev.columns:
            df_dev_x = df_dev.drop("LABEL", axis=1)
            df_dev_y = df_dev.LABEL
            Y_valid = True
        else:
            df_dev_x = df_dev
            df_dev_y = None
            Y_valid = False

        # Process the data for convolutional neural networks
        if model_ml in ["cnn", "gan"]:
            # Process dataset
            LFP = LightFluxProcessor(
                fourier=False, normalize_c=True, gaussian=True, standardize=True
            )
            df_train_x, df_dev_x = LFP.process(df_train_x, df_dev_x)
            # Rejoin X and Y
            df_train_processed = DataFrame(df_train_x).join(DataFrame(df_train_y))
            if Y_valid:
                df_dev_processed = DataFrame(df_dev_x).join(DataFrame(df_dev_y))
                # Load X and Y numpy arrays
            else:
                df_dev_processed = DataFrame(df_dev_x)

            # Save the data on temporary arrays
            X_train_tmp, Y_train_tmp = np_X_Y_from_df(df_train_processed, True)
            X_dev_tmp, Y_dev_tmp = np_X_Y_from_df(df_dev_processed, Y_valid)

            # Slice the temporary arrays and reshape
            X_train_tmp = X_train_tmp[:, :array_lenght].reshape(
                    -1, image_size, image_size, 1
            )
            X_dev_tmp = X_dev_tmp[:, :array_lenght].reshape(
                    -1, image_size, image_size, 1
            )
            # Convert labels to the required shape
            Y_train_tmp = Y_train_tmp.reshape(-1, 1)
            Y_dev_tmp = Y_dev_tmp.reshape(-1, 1)
            # Assign to class attributes directly
            self.X_train = array(X_train_tmp)
            self.Y_train = array(Y_train_tmp)
            self.X_dev = array(X_dev_tmp)
            self.Y_dev = array(Y_dev_tmp)
            #
            # Print data set stats
            (n_x, height, width, channels) = (
                self.X_train.shape
            )  # (n_x: input size)
            n_y = self.Y_train.size  # n_y : output size
            print("X_train.shape: ", self.X_train.shape)
            print("Y_train.shape: ", self.Y_train.shape)
            print("X_dev.shape: ", self.X_dev.shape)
            print("Y_dev.shape: ", self.Y_dev.shape)
            print("n_x: ", n_x)
            print("Images heigh: ", height)
            print("Images width: ", width)
            print("Number of channels: ", channels)
            print("n_y: ", n_y)
        else:
            #
            # Process dataset
            LFP = LightFluxProcessor(
                fourier=True, normalize_c=True, gaussian=True, standardize=True
            )
            df_train_x, df_dev_x = LFP.process(df_train_x, df_dev_x)
            #
            # Rejoin X and Y
            df_train_processed = DataFrame(df_train_x).join(DataFrame(df_train_y))
            if Y_valid:
                df_dev_processed = DataFrame(df_dev_x).join(DataFrame(df_dev_y))
                # Load X and Y numpy arrays
            else:
                df_dev_processed = DataFrame(df_dev_x)
            #
            self.X_train, self.Y_train = np_X_Y_from_df(df_train_processed, True)
            self.X_dev, self.Y_dev = np_X_Y_from_df(df_dev_processed, Y_valid)

            #
            # Print data set stats
            (num_examples, n_x) = (
                self.X_train.shape
            )  # (n_x: input size, m : number of examples in the data set)
            n_y = self.Y_train.size  # n_y : output size
            print("X_train.shape: ", self.X_train.shape)
            print("Y_train.shape: ", self.Y_train.shape)
            print("X_dev.shape: ", self.X_dev.shape)
            print("Y_dev.shape: ", self.Y_dev.shape)
            print("n_x: ", n_x)
            print("num_examples: ", num_examples)
            print("n_y: ", n_y)

    def get_train(self):
        """
        Return the training data

        Returns:
            Training data
        """
        return self.X_train, self.Y_train

    def get_dev(self):
        """
        Return the development data

        Returns:
            Development data
        """
        return self.X_dev, self.Y_dev
