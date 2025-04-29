# ActSOM: Exploring model activations through self-organising maps

This repository contains tools to create self-organising maps (SOMs) for exploring activations in neural network models in pytorch. In particular:
  - process_model.py creates SOMs for a model and a dataset
  - view_som.py provides different options to visualise the produced maps
  - view_freqs show the frequency map of a SOM, possibly with a concept
  - som_res visualise how the output of the model are distributed in a SOM of a layer. Works only for 1D layers.
  
The examples provided (painters and agenet) are a models based on predicting whether painters have paintings in significant museums from their bio, and on resent18 to predict the age of people from their picture, using the UTKFace dataset.