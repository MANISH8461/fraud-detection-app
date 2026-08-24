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
- Provenance beyond the Kaggle download is undocumented — see the
  `location_risk` note below for why that matters.

## Key decisions (and why)
- **Accuracy alone is misleading here.** A model that always predicts
  "not fraud" would score 93.6% accuracy while catching zero fraud.
  Evaluation is based on **precision, recall, and F1 for the fraud class**,
  using a confusion matrix — not accuracy.
- **`stratify=y`** used in every split so train/validation/test all keep
  the same 93.6/6.4 class ratio.
- **`class_weight='balanced'`** in `RandomForestClassifier` so the model
  doesn't just learn to ignore the minority (fraud) class.
- **Threshold tuned on a validation set, not the test set.** Data is split
  into train/validation/test (60/20/20). Threshold candidates (0.2–0.6)
  are evaluated only on validation; the test set is touched exactly once,
  after the threshold is already locked in, to produce the numbers below.
  An earlier version of this project tuned the threshold directly against
  the test set and reported metrics on that same set — that's leakage, and
  it made the earlier numbers optimistically biased. This version fixes it.
- **`location_risk` was investigated and deliberately excluded from the
  deployed model — see below.**

## `location_risk`: investigated, not deployed
The dataset includes a `location_risk` column (low/medium/high) not used
in the model above. I tested it: combined with `amount`, it produces
**perfect or near-perfect separation** between fraud and non-fraud — a
Random Forest, AdaBoost, and Gradient Boosting model all reached 99.5–100%
accuracy when it was included, across multiple train/validation/test splits.

That result is the opposite of reassuring. Real-world fraud signals are
almost never this clean, and I have no documentation for how this Kaggle
dataset (or `location_risk` specifically) was generated. The most likely
explanation is that the dataset is synthetic and `is_fraud` was produced
using a rule involving `amount` and `location_risk` directly — which would
make "100% accuracy" a property of the dataset's generation process, not
evidence of a working fraud detector.

Rather than deploy a model that looks unrealistically good for a reason I
can't verify, I kept `location_risk` out of the deployed model and am
documenting the finding here instead. The numbers below are lower, but I
can actually stand behind them.

## Results (test set, threshold = 0.2, selected on validation)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not Fraud (0) | 0.989 | 0.947 | 0.967 |
| Fraud (1) | 0.524 | 0.846 | 0.647 |

Trade-off: this threshold catches ~85% of fraud cases, at the cost of more
false alarms on genuine transactions.

**On sample size:** the test set contains only 13 fraud cases. A validation
run on a different 200-row split of the same data produced fraud-F1 of
0.44 instead of 0.65 at the same threshold-selection process — a ~20-point
swing from which 200 rows happened to land in the holdout. Treat the numbers
above as directionally honest, not precise to two decimal places.

## Project structure
```
fraud_app/
├── train_model_final.py  # trains the model, saves fraud_model.pkl
├── app.py                 # Flask app that serves predictions
├── fraud_model.pkl         # trained model bundle (model + threshold + features)
├── templates/
│   └── index.html         # form UI, dynamically built from model features
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
   `train_model_final.py` if it's named differently. Your CSV needs these
   columns: `amount, account_age_days, num_prev_transactions, is_fraud`

3. Train the model:
   ```
   python train_model_final.py
   ```
   This prints train/validation/test sizes, the validation threshold sweep,
   the final test-set confusion matrix + classification report, and saves
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
- Determine the actual provenance of `location_risk` (check the Kaggle
  listing directly) — if it turns out to be legitimate pre-transaction
  data, it could be added back in with real confidence.
- Try SMOTE for oversampling the minority class.
- Widen the `except Exception` block in `app.py` into specific error
  handling — right now a real bug (e.g. a bad pickle) and a genuine bad
  user input both surface as the same generic "Invalid input" message.
- Add input validation feedback on the form (min/max ranges per feature).