# ActSOM: Exploring model activations through self-organising maps

This repository contains tools to create self-organising maps (SOMs) for exploring activations in neural network models in pytorch. In particular:
  - process_model.py creates SOMs for a model and a dataset
  - view_som.py provides different options to visualise the produced maps

The example provided (agenet) is a model based on resent18 to predict the age of people from their picture, using the UTKFace dataset.