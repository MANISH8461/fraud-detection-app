# Fraud Detection — ML Model + Flask Web App

🔗 **[Try it live here](https://fraud-detection-app-x9ie.onrender.com)**

A small end-to-end project: train a fraud classifier, evaluate it honestly
(not just accuracy), and deploy it as a simple interactive web app.

> **Note:** This app runs on Render's free tier, which "sleeps" after ~15
> minutes of inactivity. The first request after sleeping can take 30-50
> seconds to wake up — this is normal, not a bug.

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
depending on business priorities (see `train_model.ipynb`).

## Project structure
```
fraud_app/
├── train_model.ipynb    # trains the model, saves fraud_model.pkl
├── app.py                # Flask app that serves predictions
├── fraud_model.pkl        # trained model bundle (model + threshold + features)
├── templates/
│   └── index.html        # form UI, dynamically built from model features
├── requirements.txt
├── Procfile
└── README.md
```

## How to run locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Put your Kaggle CSV in this folder and update the filename in
   `train_model.ipynb` if it's named differently. Your CSV needs these columns:
   `amount, account_age_days, num_prev_transactions, is_fraud`

3. Train the model by running all cells in:
   ```
   train_model.ipynb
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

1. Push this project to GitHub — including `fraud_model.pkl` (already trained).
2. Go to [render.com](https://render.com) and sign up (free tier is fine).
3. Click **New → Web Service**, connect your GitHub account, and select
   this repo.
4. Fill in the settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Click **Create Web Service**. Render installs dependencies and gives you
   a live URL like `https://your-app-name.onrender.com`.

## Possible next steps
- Try SMOTE for oversampling the minority class
- Add more features if available (transaction time, location, device, etc.)
- Add input validation feedback on the form (min/max ranges per feature)
