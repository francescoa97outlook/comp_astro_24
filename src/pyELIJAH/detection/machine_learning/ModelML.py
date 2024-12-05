from pathlib import Path
from datetime import datetime
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import tensorflow as tf
from numpy import rint, count_nonzero
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


class ModelML:
    """
        Model ML class containing the arguments and methods to
        generate ML models, train them and test them.
    """
    def __init__(
            self, output_folder, ML_type, kernel="", degree_poly=4, data_shape=100,
            n_hidden_layers=1, n_neurons=1, dropout_rate=0.05, epoch=50,
            steps_per_epoch=None, batch_size=32, kernel_size=3, pool_size=2
    ):
        """
        This init the ModelML object

        Args:
            output_folder: path of the output folder where plot and log will be saved
            ML_type: type of machine learning model [SVM, NN, CNN]
            kernel: [SVM] type of kernel used for the model [rbf, linear, polynomial]
            degree_poly: [SVM] type of polynomial used for the polynomial model
            data_shape: [NN, CNN] shape of the input data for the input layer
            n_hidden_layers: [NN, CNN] number of hidden layers (dense and dropout for NN,
             MaxPool, Conv2D for CNN) in the model
            n_neurons: [NN, CNN] number of neurons used for the hidden layers in the model
            dropout_rate: [NN, CNN] dropout rate of the Dropout layer
            epoch: [NN, CNN] number of epochs used for the model
            steps_per_epoch: [NN, CNN] number of steps per epoch used for the model
            batch_size: [NN, CNN] batch size used for the model
            kernel_size: [CNN] size of the kernel used for the Conv2D layer
            pool_size: [CNN] size of the pool used for the MaxPool2D layer
        """
        self.output_folder = output_folder
        self.ML_type = ML_type
        self.kernel = kernel
        self.degree_poly = degree_poly
        self.data_shape = data_shape
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.dropout_rate = dropout_rate
        self.proceed_with_model = True
        self.model = None
        self.epoch = epoch
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        # Get the current date and time
        current_datetime = datetime.now()
        # Format the date and time in various styles
        self.current_datetime = current_datetime.strftime("%Y_%m_%dT%H_%M_%S")

    """
        - tf.keras.layers.Input(shape)
          This layer defines the shape of the input data.
          It acts as the starting point for the model, where shape specifies
          the dimensions of the input
        - tf.keras.layers.Flatten()
          Converts multidimensional input data into a 1D array
          NOT USEFUL IN OUR CASE
        - tf.keras.layers.Dense(1, activation="relu")
          A dense layer connects every input neuron to every output neuron (first argument) .
          It applies a linear transformation followed by a non-linear
          activation function.
          Each neuron in this layer computes:
          z = w1x1 + w2x2 + ⋯ + wnxn + b
          w are learnable weights, x are inputs, b is the learnable bias term.
          The result, z, is then passed through an activation
          function (ReLU in this case) to introduce non-linearity: ReLU(z) = max(0,z)
          Sets all negative values to 0, introducing non-linearity to the model.
          This activation ensures that the layer can learn non-linear mappings,
          which are crucial for solving complex problems.
        - tf.keras.layers.Dropout(dropoutf),
           Randomly sets 50% of the neurons in the previous layer to 0 during training.
           This forces the network to learn robust patterns instead of
           memorizing specific features.
                - Overfitting occurs when the model performs well on training data but
                poorly on unseen data. Dropout helps reduce overfitting.
                - By "turning off" certain neurons, the network learns multiple
                independent representations, improving generalization.
          I.E. Without Dropout: Neuron A might dominate predictions because the model learns
          to rely too heavily on it. With Dropout: Neuron A is occasionally disabled,
          forcing other neurons to contribute to predictions.
        - tf.keras.layers.Dense(1, activation="sigmoid"),
          Output layer
          Outputs a single value, which represents the probability
          that the input belongs to the positive class.
          Uses the sigmoid activation function: sigma = 1 / (1 + e^-z)
          This maps any input zz to the range [0, 1].
          In a binary classification problem, the sigmoid ensures that
          the output can be interpreted as a probability.
          This is set to 1 because, in our case, we have only one class
          (exoplanets or not)
        - tf.keras.layers.Conv2D(filters, kernel_size),
          This layer creates a convolution kernel that is convolved with
          the layer input over a single spatial (or temporal) dimension
          to produce a tensor of outputs.
        - tf.keras.layers.MaxPool2D(pool_size),
          Downsamples the input along its spatial dimensions (height and width)
          by taking the maximum value over an input window (of size defined
          by pool_size) for each channel of the input
    """

    def build_model(self):
        """
        This function builds the model used for the machine learning detection.
        There are 3 possibilities:
            - SVM, using scikit-learn SVC https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html
            - NN, using tensorflow NN https://www.tensorflow.org/tutorials/quickstart/beginner
            - CNN, using tensorflow CNN https://www.tensorflow.org/tutorials/images/cnn
               also https://medium.com/@mayankverma05032001/binary-classification-using-convolution-neural-network-cnn-model-6e35cdf5bdbb

        """
        # Model uses support vector machine algorithm
        if self.ML_type == "svm":
            # Model creation using the SVC function
            self.model = SVC(kernel=self.kernel, degree=self.degree_poly)
        # Model uses neural networks algorithm
        elif self.ML_type == "nn":
            # Creation of the model
            self.model = tf.keras.models.Sequential()
            # Add input layer large as the size of the input data
            self.model.add(tf.keras.layers.Input((self.data_shape[1],)))
            # Add flatten layer to create mono-dimensional the data
            self.model.add(tf.keras.layers.Flatten())
            # Add a certain amount of dense and dropout layers
            for _ in range(self.n_hidden_layers):
                self.model.add(tf.keras.layers.Dense(self.n_neurons, activation="relu"))
                self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
            # Add the output layer
            self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            # Loss funciton calculation
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            # Model compilation
            self.model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        # Model uses convoluted neural networks algorithm
        elif self.ML_type == "cnn":
            # Creation of the model
            self.model = tf.keras.models.Sequential()
            # Add Conv2D layer and specify input shape
            self.model.add(tf.keras.layers.Conv2D(
                self.n_neurons, kernel_size=(self.kernel_size, self.kernel_size),
                activation="relu", input_shape=self.data_shape[1:],
                data_format="channels_last"
            ))
            # Add a maxpooling layer
            self.model.add(tf.keras.layers.MaxPool2D(
                (self.pool_size, self.pool_size), data_format="channels_last"
            ))
            # Add Dropout layer
            self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
            # Add a certain amount of convolution and maxpooling layers
            for _ in range(self.n_hidden_layers):
                self.model.add(tf.keras.layers.Conv2D(
                    self.n_neurons, kernel_size=(self.kernel_size, self.kernel_size),
                    activation="relu", data_format="channels_last"
                ))
                self.model.add(tf.keras.layers.MaxPool2D(
                    (self.pool_size, self.pool_size), data_format="channels_last"
                ))
                self.model.add(tf.keras.layers.Dropout(self.dropout_rate))

            # Add a Flatten layer
            self.model.add(tf.keras.layers.Flatten(data_format="channels_last"))
            # Add dense layer for classification
            self.model.add(tf.keras.layers.Dense(self.n_neurons, activation="relu"))
            # Add dropout layer
            self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
            # Add the output layer
            self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            # Loss funciton calculation
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # Model compilation
            self.model.compile(optimizer="adam", loss=loss_fn,
                               metrics=["accuracy"])
        # The model does not exist
        else:
            print("The model you choose is not supported.")
            self.proceed_with_model = False
            self.model = None

    def train(self, X, Y):
        """
        This function trains the model

        Args:
            X: array of elements used to train the model
            Y: labels of the training data
        """
        # If the model does not exist, end function
        if not self.proceed_with_model:
            return
        # SVM model training
        if self.ML_type == "svm":
            # Train model
            self.model.fit(X, Y)
        else:
            # NN model training
            if not self.ML_type == "cnn":
                # TODO: could this be used also for cnn?
                # Add fitted samples to the data to enlarge the dataset
                sm = SMOTE()
                X, Y = sm.fit_resample(X, Y)
            # Train model
            history = self.model.fit(X, Y.ravel(), epochs=self.epoch,
                                     batch_size=self.batch_size,
                                     steps_per_epoch=self.steps_per_epoch)
            # Plot accuracy of the model
            output_plot = str(Path(self.output_folder,
                                   f"plot_{self.ML_type}_{self.current_datetime}.png"
                                   ))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
            plot_title = f"\nLayers: {self.n_hidden_layers}, neurons: {self.n_neurons}, Dropout rate: {self.dropout_rate}"
            fig.suptitle(plot_title)
            # list all data in history
            print(history.history.keys())
            # summarize history for accuracy
            ax1.plot(history.history["accuracy"])
            ax1.set_title("Model Accuracy")
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel("Epoch")
            ax1.legend(["train", "test"], loc="upper left")
            #
            # summarize history for loss
            ax2.plot(history.history["loss"])
            ax2.set_title("Model Loss")
            ax2.set_ylabel("loss")
            ax2.set_xlabel("epoch")
            ax2.legend(["train", "test"], loc="upper left")
            plt.savefig(output_plot)
            plt.close(fig)

    def predict(self, X, Y_results=None, label=""):
        """
        This function predicts the class labels for the given data

        Args:
            X: array of elements used to derive the label from the trained model
            Y_results: array of labels to che k the model results, if provided
        """
        # If the model does not exist, end function
        if not self.proceed_with_model:
            return
        # Start of the log string
        log_temp = (
            f"--------------------------------------"
            f"\n{self.ML_type} model"
        )
        # Predict of the data for SVM model
        if self.ML_type == "svm":
            predicted = self.model.predict(X)
            log_temp += (
                f"\n--------------------------------------"
                f"\nKernel: {self.kernel}"
                f"\nDegree (useful only if poly kernel): {self.degree_poly}"
            )
        # Predict of the data for NN and CNN model
        else:
            predicted = self.model.predict(X, batch_size=self.batch_size)
            log_temp += (
                f"\n--------------------------------------"
                f"\nNumber of dense and dropout layers: {self.n_hidden_layers}"
                f"\nNumber of neurons: {self.n_neurons}"
                f"\nDropout rate: {self.dropout_rate}"
                f"\nEpochs: {self.epoch}"
                f"\nBatch size: {self.batch_size}"
            )
            if self.ML_type == "cnn":
                log_temp += (
                    f"\nKernel size: {self.kernel_size}"
                    f"\nPooling size: {self.pool_size}"
                )
        #
        predicted = rint(predicted)
        # If results label are provided, it creates a statistics of precision and confusion matrix
        if Y_results is not None:
            accuracy_train = accuracy_score(Y_results, predicted)
            # Display the prediction results
            precision_train = precision_score(Y_results, predicted)
            recall_train = recall_score(Y_results, predicted)
            confusion_matrix_train = confusion_matrix(Y_results, predicted)
            log_temp += (
                f"\n--------------------------------------"
                f"\nError: {1.0 - accuracy_train}"
                f"\n------------"
                f"\nPrecision: {precision_train}"
                f"\n------------"
                f"\nConfusion Matrix:\n{confusion_matrix_train}"
                f"\n------------"
                f"\nPositive Predictions: {count_nonzero(predicted)}"
                f"\n------------"
                f"\nRecall: {recall_train}"
                f"\n--------------------------------------"
            )
        else:
            log_temp += (
                f"\n--------------------------------------"
                "No Y results provided, the error, precision, "
                "confusion matrix and recall cannot be calculated"
                f"\n--------------------------------------"
            )
        # Print of the log and store in the log file
        print(log_temp)
        output_log = str(Path(self.output_folder, f"log_{self.ML_type}{label}_{self.current_datetime}.txt"))
        with open(output_log, "w") as file:
            file.write(log_temp)
