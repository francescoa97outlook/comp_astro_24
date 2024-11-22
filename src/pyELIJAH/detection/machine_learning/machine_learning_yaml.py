import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from numpy import random

from pyELIJAH.detection.machine_learning.Dataset import DatasetTrainDev
from pyELIJAH.detection.machine_learning.ModelML import ModelML


def machine_learning(
    input_data_folder, output_data_folder, filename_train, filename_dev,
    ml_model, params, render_plot=False,
):
    """
    Machine Learning function used to detect exoplanets

    Args:
        input_data_folder (str): Path to the folder containing the training data
        output_data_folder (str): Path to the folder where to save the output data
        filename_train (str): Name of the csv file containing the training data
        filename_dev (str): Name of the csv file containing the development data
        ml_model (str): Machine Learning model decided
        params (parameters obj): Object containing model parameters
        render_plot (bool): If true, render the plot for NN models
    """
    random.seed(1)
    # Retrieve data information
    data_object = DatasetTrainDev(input_data_folder, filename_train, filename_dev)
    X_train, Y_train = data_object.get_train()
    X_dev, Y_dev = data_object.get_dev()
    # Build model
    ml_object = ModelML(
        output_data_folder, ml_model, params.get("kernel"), params.get("degree_poly"),
        X_dev[0, :].shape, params.get("n_hidden_layers"), params.get("n_neurons"),
        params.get("dropout_rate"), params.get("epoch"), params.get("batch_size"),
    )
    ml_object.build_model()
    # Train model
    ml_object.train(X_train, Y_train, render_plot)
    # Predict model
    ml_object.predict(X_dev, Y_dev)
