
This repository contains code to analyse neural network activations agains concepts using self-organising maps (SOMs), as described in the paper "inding Concept Representations
in Neural Networks with Self-Organizing Maps". That process takes as input a trained model, the dataset on which it was trained, and concept annotations for that concept.
The code to build this and configure the process on the two case studies presented in the paper is also available in this repository.

# Overview of the process

 1. `actsom.py` contains the definition of the `ActSom` class which is used to train, populate and assess SOMs
 2. `minisom.py` is a copy of the minisom library, with a small change to avoid heavy computation in debug mode
 3. `create_base_soms.py` is a script that creates SOMs trained and populated on the entire dataset for each layer of the network. It takes as input (command line argument) a config file in JSON including paths to the model and the activation vectors on the dataset.
 4. `create_concept_soms.py` is a script that populate base SOMs with subsets of the dataset corresponding to concepts. It takes as input (command line argument) a config file in JSON including paths to the base SOMs, the activation vectors, and the concept annotatations.
 5. `view_som.py` is used to display a som. It takes as input (command line argument) a SOM file (from steps 3 and 4), or a directory where to find a set of SOM files and one where to save images of those SOMs.

