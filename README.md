# PRED_490_BIGSHAPES
# The code and the files in this repository train a MLP Neural Net with 1 hidden layer
#
# The python code does the following:
# Read training data from 3 training files (change numTrainingFiles in the code to add more)
# the following are provided:
# ABCD_26_VARS0.csv - first variant
# ABCD_26_VARS1.csv - second variant
# ABCD_26_VARS2.csv - third variant
#
# The NN then reads the class map file ABCD_26_VARS_classes.csv
# This maps the outputs to the "big shapes" defined in the mapping
#
# The NN accepts the following inputs: alpha, eta, noise % (gamma), number of hidden nodes
#
# The output is written to a csv file whose name contains the coded prefix "ABCD_VARS_26" and then the following separated by '_' number of hidden nodes, ahlpa, eta, epsilon, max iterations and noise. The example name is ABCD_26_VARS_8_1_1_0.01_20000_0.15
