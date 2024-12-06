# pyELIJAH
pyELIJAH:
(py)thon (E)xoplanets (L)ight-curve in (I)nfrared, not (J)ust (A)tmospheres. The (H) is silent :)

Using the pyELIJAH command, the user can choose between different arguments to be passed.
- pyelijah -h, --help: shows the help message
- pyelijah -i INPUT_FILE, --input INPUT_FILE: a yaml file containing three list, one for each main modules. Each list contains yaml_files used in the corresponding module.
- pyelijah -t, --transit: select the transit method
- pyelijah -id DIRECTORY_INPUT, --directory_input DIRECTORY_INPUT: Directory of the input files (yaml, limb darkening...). If the user wants to set another folder to get their input data, use this option. In this case, just put the absolute path of the folder. Otherwise, put their data inside the Data folder (default one) and do not use this option.
- pyelijah -res DIRECTORY_RESULTS, --directory_results DIRECTORY_RESULTS: directory where to store the results. If the user wants to set another folder to store their output data, use this option. In this case, just put the absolute path of the folder. Otherwise, the output will be stored in the Results folder.
- pyelijah -multi, --multi_plot: flag to plot all planets information singularly
- pyelijah -d, --detect: Command that is used to accept a limited set of strings concerning the machine learning algorithm used to detect exoplanets in a set of data. The accepted values are: "svm" for support vector machine, "nn" for neural network, "cnn" for convolutional neural network.
- pyelijah -dr, --dream: used to call the GAN architecture that allows to 'dream' a new exoplanetary transit light curve
- pyelijah -a, --atmosphere: atmospheric characterisation from input transmission spectrum
