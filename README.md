# Fraud Detection — ML Model + Flask Web App

A small end-to-end project: train a fraud classifier, evaluate it honestly
(not just accuracy), and deploy it as a simple interactive web app.

## Problem
Given a transaction's `amount`, `account_age_days`, and `num_prev_transactions`,
predict whether it's fraudulent.

## Dataset
- Source: Kaggle
- 1,000 rows, **imbalanced**: 936 non-fraud (93.6%) vs 64 fraud (6.4%)

## Key decisions (and why)
- **Accuracy alone is misleading here.** A model that always predicts
  "not fraud" would score 93.6% accuracy while catching zero fraud. Evaluation
  is based on **precision, recall, and F1 for the fraud class**, using a
  confusion matrix — not accuracy.
- **`stratify=y`** used in the train/test split so both sets keep the same
  93.6/6.4 class ratio.
- **`class_weight='balanced'`** in `RandomForestClassifier` so the model
  doesn't just learn to ignore the minority (fraud) class.
- **Decision threshold tuned** (tested 0.2 / 0.3 / 0.4 / 0.5) instead of
  using the default 0.5 cutoff. 0.3 gave the best F1-score balance between
  catching fraud (recall) and avoiding false alarms (precision).
- **EDA before modeling:** checked the correlation heatmap, then verified
  `amount` was a genuinely strong signal (not a leakage artifact) by plotting
  its distribution split by fraud/non-fraud (histogram + boxplot, log scale).

## Results (test set, threshold = 0.3)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not Fraud (0) | 0.99 | 0.94 | 0.96 |
| Fraud (1) | 0.50 | 0.85 | 0.63 |

Trade-off: this threshold catches ~85% of fraud cases, at the cost of more
false alarms on genuine transactions. The threshold is easy to change
depending on business priorities (see `train_model.py`).

## Project structure
```
fraud_app/
├── train_model.py       # trains the model, saves fraud_model.pkl
├── app.py                # Flask app that serves predictions
├── templates/
│   └── index.html        # simple form UI
├── requirements.txt
└── README.md
```

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Put your Kaggle CSV in this folder and update the filename in
   `train_model.py` (`DATA_PATH = "fraud_dataset.csv"`) if it's named
   differently. Your CSV needs these columns:
   `amount, account_age_days, num_prev_transactions, is_fraud`

3. Train the model:
   ```
   python train_model.py
   ```
   This prints the confusion matrix + classification report, and saves
   `fraud_model.pkl`.

4. Run the web app:
   ```
   python app.py
   ```
   Open **http://127.0.0.1:5000** in your browser, enter transaction
   details, and get a Fraud / Not Fraud prediction with probability.

## Deploying to Render (free live link)

1. **Push this project to GitHub** — create a new repo, add all these files
   (including `fraud_model.pkl` once you've run `train_model.py` — or set up
   Render to run the training step, see note below), commit, and push.

2. **Go to [render.com](https://render.com)** and sign up (free tier is fine).

3. Click **New → Web Service**, connect your GitHub account, and select
   this repo.

4. Fill in the settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already set in the `Procfile`,
     Render usually auto-detects this)

5. Click **Create Web Service**. Render will install dependencies, run the
   build, and give you a live URL like `https://your-app-name.onrender.com`.

6. Open that link — your form should be live for anyone to try.

**Important note on the model file:** `fraud_model.pkl` needs to exist in the
repo for `app.py` to load it. Either:
- Run `train_model.py` locally first and commit `fraud_model.pkl` to GitHub
  along with your code (simplest — do this), **or**
- Add a build step that runs `train_model.py` before starting the app (more
  advanced, only needed if you don't want to commit the `.pkl` file).

**Free tier note:** Render's free web services "sleep" after ~15 minutes of
inactivity and take ~30-50 seconds to wake up on the next request. This is
normal — just a heads up so it doesn't look broken when you demo it.

## Possible next steps
- Try SMOTE for oversampling the minority class
- Add more features if available (transaction time, location, device, etc.)
- Deploy publicly (Render / Railway / Hugging Face Spaces) for a live demo link
