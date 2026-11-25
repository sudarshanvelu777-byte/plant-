# model/

This folder holds the artifacts produced by `model_training.ipynb`:

- `plant_disease_model.h5` — the trained Keras CNN (not included here; train it
  in Colab and download it into this folder).
- `class_indices.json` — maps the model's output index to a PlantVillage class
  label (generated automatically by the notebook).
- `disease_info.json` — maps each class label to a human-readable description
  and prevention/treatment steps shown in the UI. A starter version with a
  few common classes filled in is already in this folder — extend it with
  entries for every class in your trained model (`app.py` falls back to a
  generic message for any label not listed).
