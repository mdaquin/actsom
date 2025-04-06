# ActSOM: Exploring model activations through self-organising maps

This repository contains tools to create self-organising maps (SOMs) for exploring activations in neural network models in pytorch. In particular:
  - process_model.py creates SOMs for a model and a dataset
  - view_som.py provides different options to visualise the produced maps
  - run_dataset.py compute the results of runing the whole dataset through the SOMs of each layer, producing a result file (csv) with the target, prediction, concepts, and the cells of the som in which the data point falls. It also produces a file for each layer of the activations of the hidden layer of the sparse autoencoder.
  - test_concept computes metrics for each layer for a given concept.
  - view_freqs show the frequency map of a SOM, possibly with a concept
  - view_metrics display the metrics.
  - view_sae.py displays the average, min and max activations of the encoding layer of the sparse autoencoder. If a concept is given, it distinguishes between data points in the concept and those out of it. (this part needs to be made more efficient, by adding the concept in the same file so we don't have to iterate and concatenate over it). A filter can be given to only show the units having a significant difference (percent diff of the average).
  - find_concept_sae.py show the relation between concepts and encoding units in sae 
  
The examples provided (painters and agenet) are a models based on predicting whether painters have paintings in significant museums from their bio, and on resent18 to predict the age of people from their picture, using the UTKFace dataset.