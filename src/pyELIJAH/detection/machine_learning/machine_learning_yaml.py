import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from numpy import random

from pyELIJAH.detection.machine_learning.Dataset import DatasetTrainDev
from pyELIJAH.detection.machine_learning.ModelML import ModelML


def machine_learning(
    input_data_folder, output_data_folder, ml_model, params_list
):
    """
    Machine Learning function used to detect exoplanets

    Args:
        input_data_folder (str): Path to the folder containing the training data
        output_data_folder (str): Path to the folder where to save the output data
        params_list (list of parameters obj): Object containing list of model parameters
        ml_model (str): Machine Learning model decided
    """
    random.seed(1)
    for params in params_list:
        # Retrieve data information
        data_object = DatasetTrainDev(
            input_data_folder, params.get("filename_train"),
            params.get("filename_dev"), ml_model
        )
        X_train, Y_train = data_object.get_train()
        X_dev, Y_dev = data_object.get_dev()
        # Build model
        ml_object = ModelML(
            output_data_folder, ml_model, params.get("kernel"), params.get("degree_poly"),
            X_train.shape, params.get("n_hidden_layers"), params.get("n_neurons"),
            params.get("dropout_rate"), params.get("epoch"), params.get("batch_size"),
        )
        ml_object.build_model()
        # Train model
        ml_object.train(X_train, Y_train)
        # Predict model
        print("Prediction TRAIN...")
        ml_object.predict(X_train, Y_train)
        print("Prediction DEV...")
        ml_object.predict(X_dev, Y_dev)
