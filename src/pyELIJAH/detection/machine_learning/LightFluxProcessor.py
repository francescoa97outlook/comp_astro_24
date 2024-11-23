from pandas import DataFrame
from numpy import zeros, abs
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler


def fourier_transform(X):
    """
    Applies Fourier transform to X

    Args:
        X: array to apply Fourier transform
    """
    return abs(fft.fft(X.values, n=X.size))


class LightFluxProcessor:
    """
    Class that pre-process the data
    """
    def __init__(self, fourier=True, normalize_c=True, gaussian=True, standardize=True):
        """
        Constructor

        Args:
            fourier: whether to apply Fourier transform to work on the frequency domain
            normalize_c: whether to normalize the data
            gaussian: whether to apply Gaussian filter to smooth the data
            standardize: whether to standardize the data to balance the weight of the data
             in case there is a majority of elements appertaining to only one class
        """
        self.fourier = fourier
        self.normalize = normalize_c
        self.gaussian = gaussian
        self.standardize = standardize

    def process(self, df_train_x_f, df_dev_x_f):
        """
        Effective pre-processing of data

        Args:
            df_train_x_f: dataframe containing training data
            df_dev_x_f: dataframe containing development data
        """
        # Apply fourier transform
        if self.fourier:
            print("Applying Fourier...")
            shape_train = df_train_x_f.shape
            shape_dev = df_dev_x_f.shape
            df_train_x_f = df_train_x_f.apply(fourier_transform, axis=1)
            df_dev_x_f = df_dev_x_f.apply(fourier_transform, axis=1)
            df_train_x_build = zeros(shape_train)
            df_dev_x_build = zeros(shape_dev)
            for ii_f, x in enumerate(df_train_x_f):
                df_train_x_build[ii_f] = x
            for ii_f, x in enumerate(df_dev_x_f):
                df_dev_x_build[ii_f] = x
            df_train_x_f = DataFrame(df_train_x_build)
            df_dev_x_f = DataFrame(df_dev_x_build)
            # Keep first half of data as it is symmetrical after previous steps
            df_train_x_f = df_train_x_f.iloc[:, :(df_train_x_f.shape[1] // 2)].values
            df_dev_x_f = df_dev_x_f.iloc[:, :(df_dev_x_f.shape[1] // 2)].values

        # Normalize
        if self.normalize:
            print("Normalizing...")
            df_train_x_f = DataFrame(normalize(df_train_x_f))
            df_dev_x_f = DataFrame(normalize(df_dev_x_f))
            # df_train_x = df_train_x.div(df_train_x.sum(axis=1), axis=0)
            # df_dev_x = df_dev_x.div(df_dev_x.sum(axis=1), axis=0)

        # Gaussian filter to smooth out data
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x_f = ndimage.gaussian_filter(df_train_x_f, sigma=10)
            df_dev_x_f = ndimage.gaussian_filter(df_dev_x_f, sigma=10)

        if self.standardize:
            # Standardize X data
            print("Standardizing...")
            std_scaler = StandardScaler()
            df_train_x_f = std_scaler.fit_transform(df_train_x_f)
            df_dev_x_f = std_scaler.transform(df_dev_x_f)

        print("Finished Processing!")
        return df_train_x_f, df_dev_x_f