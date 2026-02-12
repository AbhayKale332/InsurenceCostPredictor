# Insurance Price Predictor — Linear Regression (beginner project) 🚑💡

## 🌐 Live Deployment
🔗 https://insurancecostpredictorby.akashchaudhari.in

**Short overview**

This project shows a complete, beginner-friendly workflow for predicting insurance charges using a Linear Regression model. The `InsurenceHypothesis.ipynb` notebook explains each step slowly: EDA, preprocessing (encoding + log transform), model training, evaluation, and model saving.

---

## What’s included 🔍

- `InsurenceHypothesis.ipynb` — notebook with step-by-step explanations and plots.
- `Data/InsuranceData.csv` — dataset used in the notebook.
- `insurance_model.pkl` and `model_columns.pkl` — example saved model and column list (created by the notebook).

---

## Key concepts you’ll learn 🎯

- Encoding categorical features (`sex`, `smoker`, `region`) appropriately.
- Handling skewed targets with `log1p` and converting predictions back with `expm1`.
- Train/test split, model training, evaluation (R², MSE, MAE), and residual analysis.
- Saving and loading a trained model with `pickle` for simple deployment/testing.

---

## Quick start — run the notebook ✅

1. (Optional) Create and activate a Python virtual environment:

   ```powershell
   python -m venv .venv
   & .venv\Scripts\Activate.ps1
   ```

2. Install required packages:

   ```powershell
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Open and run `InsurenceHypothesis.ipynb` in VS Code / Jupyter. Run cells top-to-bottom and read the markdown explanations.

---

## Example — load saved model and predict (Python)

```python
import pickle
import numpy as np

model = pickle.load(open('insurance_model.pkl', 'rb'))
model_cols = pickle.load(open('model_columns.pkl', 'rb'))

# create an input row using the saved column order
x = np.zeros(len(model_cols))
# set features: age, sex (0=m), bmi, children, smoker (0=no), region_southeast=1, etc.
# example: 30-year male, bmi=25, 1 child, non-smoker, southeast
# set indices according to model_columns.pkl and then:
# predicted_log = model.predict([x])
# predicted_charge = np.expm1(predicted_log)
```

---

## Next steps / experiments 🚀

- Try Ridge/Lasso regularization (reduce overfitting).  
- Add interaction terms (age*bmi) or polynomial features for non-linearity.  
- Compare with tree-based models (RandomForest, XGBoost).  
- Calibrate error: report MAE in dollars for clearer interpretation.

---

## Notes for learners ✏️

- The notebook stores predictions on the log scale — always convert back with `np.expm1` before interpreting dollar values.  
- Read each markdown cell in `InsurenceHypothesis.ipynb` — the notebook intentionally explains changes step-by-step.

---

Want me to add a short `requirements.txt`, improve model performance (step-by-step), or create a small demo script that loads the model and predicts for user input? Reply which task you want next.
