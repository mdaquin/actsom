# ActSOM: Exploring model activations through self-organising maps

This repository contains tools to create self-organising maps (SOMs) for exploring activations in neural network models in pytorch. In particular:
  - process_model.py creates SOMs for a model and a dataset
  - view_som.py provides different options to visualise the produced maps
  - run_dataset.py compute the results of runing the whole dataset through the SOMs of each layer, producing a result file (csv) with the target, prediction, concepts, and the cells of the som in which the data point falls.
  - test_concept computes metrics for each layer for a given concept.
  - view_freqs show the frequency map of a SOM, possibly with a concept
  - view_metrics display the metrics.

The example provided (agenet) is a model based on resent18 to predict the age of people from their picture, using the UTKFace dataset.