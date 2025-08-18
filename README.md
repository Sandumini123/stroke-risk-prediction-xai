# Stroke Risk Prediction Using Explainable AI

This project develops a personalized **stroke risk prediction framework** using **XGBoost, SHAP, and LIME**.  
It handles class imbalance and provides **patient-specific explanations** with an interactive Streamlit app.

## 📖 Research Details
- **Title:** An Explainable Machine Learning Framework for Personalized Stroke Risk Prediction
- **Dataset:** [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
- **Methods:** Data preprocessing, class imbalance handling (Class Weight, SMOTE, ADASYN), XGBoost with randomized hyperparameter tuning
- **Explainability:** SHAP & LIME for global and local feature interpretation
- **Output:** Probability-based risk levels (Low, Moderate, High) + Patient-specific insights

## 📊 Model Performance
| Method        | PR-AUC | F1-Score | ROC-AUC |
|---------------|--------|----------|---------|
| ADASYN        | 0.1841 | 0.2273   | 0.7829  |
| SMOTE         | 0.1643 | 0.1935   | 0.7843  |
| Class-Weight  | **0.2043** | **0.2478** | **0.8037** |

## 🏥 Clinical Relevance
The system supports **real-time risk prediction** and improves clinician trust by providing:
- Risk probabilities & levels
- Patient-specific SHAP and LIME explanations
- Interactive decision-support interface

## 📂 Project Structure
```bash
├── app/
│   ├── StrokeRiskPredictor.py       # Real-time prediction interface
│   ├── model/                       # Trained model and weights
│   └── explainability/             # SHAP and LIME explainers
├── data/
│   └── stroke_data.csv             # Preprocessed dataset
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory data analysis
│   ├── Model_Training.ipynb        # Model training and evaluation
│   └── Explainability.ipynb        # SHAP & LIME explainability
├── README.md
└── requirements.txt
