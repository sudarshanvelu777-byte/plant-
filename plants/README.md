# LeafScan AI — Plant Leaf Disease Detection Website

A full-stack web app: upload a photo of a plant leaf, and a CNN trained on the
**PlantVillage** dataset predicts the disease (or confirms the plant is
healthy), with a confidence score and treatment/prevention recommendations.

**Theme:** Deep Purple Glow — dark violet background, glowing card shadows,
orange CTA accents.

---

## 1. Project structure

```
plant-disease-detector/
├── app.py                     # Flask backend — preprocessing + prediction API
├── model_training.ipynb       # Google Colab notebook — trains the CNN
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html             # Upload form + results UI
├── static/
│   └── style.css              # Deep Purple Glow styling
├── model/
│   ├── plant_disease_model.h5 # Trained model (you add this — see step 3)
│   ├── class_indices.json     # Index → class label map (you add this)
│   ├── disease_info.json      # Disease descriptions + prevention steps (starter included)
│   └── README.md
└── uploads/                   # Scratch folder (optional, for saving uploads)
```

## 2. Local setup

Requires Python 3.9–3.11.

```bash
# from the project root
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Train the model (Google Colab)

The web app needs a trained model before predictions will work.

1. Open `model_training.ipynb` in [Google Colab](https://colab.research.google.com/)
   (upload the file, or open it from Drive/GitHub).
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom:
   - Installs TensorFlow and downloads the **PlantVillage** dataset from
     Kaggle (you'll need a free Kaggle account and API token — see the
     notebook's instructions, or mount Google Drive if you already have a
     local copy of the dataset).
   - Builds a CNN using **transfer learning on MobileNetV2** and trains it
     with data augmentation.
   - Evaluates accuracy on a held-out validation split.
   - Exports three files: `plant_disease_model.h5`, `class_indices.json`,
     and a starter `disease_info.json` (with `TODO` placeholders for every
     class the model learned).
4. Download those three files and place them in this project's `model/`
   folder, overwriting the starter `disease_info.json` if you want the
   auto-generated full version — or merge your real descriptions into the
   starter file already provided (which has several common diseases
   pre-filled as examples).

> Training end-to-end on the full PlantVillage dataset (~54,000 images,
> 38 classes) takes roughly 20–40 minutes on a Colab T4 GPU with the settings
> in the notebook.

## 4. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- Drag and drop a leaf photo, or click to browse.
- Click **Analyze Leaf**.
- The app shows the uploaded image, detected disease (or "Healthy"),
  confidence percentage, a description, and prevention/treatment steps.

## 5. How it works

1. **Frontend** (`templates/index.html` + `static/style.css`): a drag-and-drop
   upload form posts the image as `multipart/form-data` to `/predict`, shows
   a loading spinner while waiting, then renders the JSON response into a
   result card.
2. **Backend** (`app.py`):
   - `preprocess_image()` opens the upload with **Pillow**, converts it to
     RGB, resizes it to 224×224 (matching training), and uses **NumPy** to
     normalize pixel values to `[0, 1]` and add a batch dimension.
   - The trained **Keras** model (`model/plant_disease_model.h5`) predicts
     class probabilities; `argmax` picks the top class and its confidence.
   - `class_indices.json` converts the predicted index back to a readable
     label (e.g. `Tomato___Early_blight` → "Tomato - Early Blight").
   - `disease_info.json` supplies the description and prevention steps for
     that label.
   - The `/predict` endpoint returns:
     ```json
     {
       "disease_name": "Tomato - Early Blight",
       "confidence": 94.21,
       "is_healthy": false,
       "description": "...",
       "prevention_steps": ["...", "...", "..."]
     }
     ```
3. **Error handling**: invalid file types, oversized files (>8 MB), corrupt
   images, and a missing/untrained model all return a clear JSON `error`
   message with an appropriate HTTP status code, which the frontend displays.

## 6. Extending it

- Add more entries to `model/disease_info.json` for every class your model
  was trained on (the Colab notebook auto-generates a `TODO` template for
  all of them to save typing).
- Swap MobileNetV2 for a different base model (e.g. EfficientNetB0) in the
  notebook for a potential accuracy/speed trade-off.
- Deploy the Flask app with Gunicorn + a platform like Render, Railway, or
  Google Cloud Run for a public URL.
- Add a history/log of past predictions using SQLite if you want users to
  review previous scans.

## Tech stack

Python · NumPy · Pillow · PlantVillage dataset · TensorFlow/Keras (trained
in Google Colab) · Flask · HTML/CSS · developed in VS Code.
