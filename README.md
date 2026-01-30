# Credit Risk Intelligence: Neural Networks vs. XGBoost

An end-to-end machine learning pipeline to predict credit card default risk using the UCI Credit Card dataset. This project demonstrates the "Model Pivot"—starting with Deep Learning and optimizing with Gradient Boosting to achieve superior results on tabular data.

## 🚀 Key Achievements
- **Model Comparison:** Benchmarked a PyTorch MLP against an optimized XGBoost model.
- **Improved Performance:** Increased ROC-AUC from **0.72 (NN)** to **0.78 (XGBoost)**.
- **Feature Engineering:** Developed "Utilization" and "Payment" ratios that improved model sensitivity.

## 📊 Results Summary
XGBoost outperformed the Neural Network across all key classification metrics, particularly in handling the minority class (defaults).

| Metric | Neural Network | XGBoost (Final) |
| :--- | :--- | :--- |
| **Accuracy** | 79.2% | **79.5%** |
| **ROC-AUC** | 0.720 | **0.781** |
| **F1-Score** | 0.520 | **0.550** |

### Visual Analysis
#### Feature Importance

The model relies heavily on repayment status (`PAY_0`) and the custom `utilization_ratio`.

#### ROC Curve

The higher curve for XGBoost indicates a much stronger ability to separate default vs. non-default cases.

## 📂 Repository Structure
- `notebooks/Neural_Net_Credit_Card_Clients_Default_Predictor.ipynb`: Initial PyTorch Multi-Layer Perceptron.
- `notebooks/XGBoost_Credit_Card_Clients_Default_Predictor.ipynb`: Feature engineering and comparison.
- `images/`: PNG files of all performance plots.
- `requirements.txt`: Python environment dependencies.

## 🛠️ Installation
```bash
pip install -r requirements.txt
