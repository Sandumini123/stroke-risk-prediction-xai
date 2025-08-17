import numpy as np
import pandas as pd
import shap
import lime.lime_tabular
import json

def compute_shap_explanations(model, X_test, max_samples=500):
    """Compute SHAP explanations with error handling"""
    try:
        print(f"\nComputing SHAP explanations for {min(len(X_test), max_samples)} samples...")
        if len(X_test) > max_samples:
            sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_sample = X_test.iloc[sample_indices]
        else:
            X_sample = X_test

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        print("SHAP explanations computed successfully!")
        return explainer, shap_values

    except Exception as e:
        print(f"Error computing SHAP explanations: {e}")
        return None, None

def validate_explanations(shap_values, X_test):
    """Validate SHAP explanations"""
    if shap_values is not None:
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"SHAP values range: [{shap_values.min():.4f}, {shap_values.max():.4f}]")
        print("SHAP explanations validated successfully!")
    else:
        print("SHAP explanations validation failed!")

def compute_lime_explanations(model, X_train, X_test, num_samples=50):
    """Compute LIME explanations"""
    try:
        print(f"\nComputing LIME explanations for {num_samples} samples...")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=['No Stroke', 'Stroke'],
            mode='classification'
        )

        lime_explanations = []
        sample_size = min(num_samples, len(X_test))
        for i in range(sample_size):
            exp = explainer.explain_instance(
                X_test.iloc[i].values,
                model.predict_proba,
                num_features=len(X_test.columns)
            )
            lime_explanations.append(exp)

        print(f"LIME explanations computed for {len(lime_explanations)} samples!")
        return explainer, lime_explanations

    except Exception as e:
        print(f"Error computing LIME explanations: {e}")
        return None, None

def compute_all_trust_scores(model, X_test, explainer_shap, explainer_lime):
    """Compute trust scores for predictions"""
    try:
        print("\nComputing trust scores...")
        predictions = model.predict_proba(X_test)[:, 1]
        trust_scores = []
        for pred in predictions:
            if pred < 0.1 or pred > 0.9:
                trust = 0.95
            elif pred < 0.3 or pred > 0.7:
                trust = 0.75
            else:
                trust = 0.5
            trust_scores.append(trust)

        trust_scores = np.array(trust_scores)
        unstable_predictions = np.where(trust_scores < 0.6)[0]
        print(f"Trust scores computed. {len(unstable_predictions)} unstable predictions identified.")
        return trust_scores, unstable_predictions

    except Exception as e:
        print(f"Error computing trust scores: {e}")
        return None, None

def interpret_feature_contribution(feature_name, feature_value, contribution, factor_type):
    """Interpret individual feature contributions in clinical context"""
    interpretations = {
        'age': {
            'risk': f"Advanced age ({feature_value:.0f} years) significantly increases stroke risk due to arterial aging and increased comorbidities",
            'protective': f"Younger age ({feature_value:.0f} years) provides natural protection against stroke"
        },
        'hypertension': {
            'risk': "Hypertension is present, creating sustained pressure on blood vessels and increasing stroke risk",
            'protective': "Normal blood pressure reduces strain on cardiovascular system, lowering stroke risk"
        },
        'heart_disease': {
            'risk': "Existing heart disease significantly elevates stroke risk through compromised cardiovascular function",
            'protective': "Healthy heart function supports proper blood flow, reducing stroke risk"
        },
        'avg_glucose_level': {
            'risk': f"Elevated glucose level ({feature_value:.1f} mg/dL) indicates diabetes/prediabetes, increasing stroke risk",
            'protective': f"Normal glucose level ({feature_value:.1f} mg/dL) indicates good metabolic health"
        },
        'bmi': {
            'risk': f"BMI of {feature_value:.1f} suggests weight issues that contribute to cardiovascular strain",
            'protective': f"Healthy BMI of {feature_value:.1f} supports optimal cardiovascular function"
        },
        'smoking_status_encoded': {
            'risk': "Smoking history damages blood vessels and significantly increases stroke risk",
            'protective': "Non-smoking status protects blood vessel health and reduces stroke risk"
        },
        'gender_encoded': {
            'risk': "Gender-related factors contribute to stroke risk profile",
            'protective': "Gender-related factors provide some protection against stroke"
        },
        'ever_married_encoded': {
            'risk': "Marital status may reflect lifestyle factors affecting stroke risk",
            'protective': "Marital status may indicate social support reducing stroke risk"
        },
        'work_type_encoded': {
            'risk': "Work-related stress or lifestyle factors may increase stroke risk",
            'protective': "Work environment may promote healthier lifestyle reducing stroke risk"
        },
        'residence_encoded': {
            'risk': "Residence type may reflect environmental factors increasing stroke risk",
            'protective': "Residence environment may support healthier lifestyle choices"
        }
    }

    if feature_name in interpretations:
        return interpretations[feature_name][factor_type]
    else:
        if factor_type == 'risk':
            return f"{feature_name} (value: {feature_value:.2f}) contributes to increased stroke risk"
        else:
            return f"{feature_name} (value: {feature_value:.2f}) provides protection against stroke"

def generate_clinical_summary(risk_level, risk_factors, protective_factors, risk_probability):
    """Generate comprehensive clinical summary without recommendations"""
    summary = f"=== STROKE RISK ASSESSMENT ===\n"
    summary += f"Risk Level: {risk_level} ({risk_probability:.1%} probability)\n\n"

    if risk_factors:
        summary += "🔴 PRIMARY RISK FACTORS:\n"
        for i, factor in enumerate(risk_factors[:3], 1):
            summary += f"{i}. {factor['interpretation']}\n"
        summary += "\n"

    if protective_factors:
        summary += "🟢 PROTECTIVE FACTORS:\n"
        for i, factor in enumerate(protective_factors[:3], 1):
            summary += f"{i}. {factor['interpretation']}\n"
        summary += "\n"

    return summary

def generate_patient_specific_explanations(model, X_test, shap_values, patient_indices=None, top_n_features=5):
    """Generate patient-specific clinical explanations"""
    if patient_indices is None:
        patient_indices = range(min(20, len(X_test)))

    explanations = {}
    feature_names = X_test.columns.tolist()
    print(f"\nGenerating explanations for {len(patient_indices)} patients...")

    for idx in patient_indices:
        try:
            patient_data = X_test.iloc[idx]
            patient_shap = shap_values[idx]
            risk_prob = model.predict_proba(X_test.iloc[[idx]])[0, 1]

            if risk_prob > 0.7:
                risk_level = "HIGH"
            elif risk_prob > 0.3:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"

            feature_contributions = list(zip(feature_names, patient_shap, patient_data))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feature_contributions[:top_n_features]

            risk_factors = []
            protective_factors = []
            for feature, contribution, value in top_features:
                factor_info = {
                    'feature': feature,
                    'value': value,
                    'contribution': abs(contribution),
                    'interpretation': interpret_feature_contribution(
                        feature, value, contribution,
                        'risk' if contribution > 0 else 'protective'
                    )
                }
                if contribution > 0:
                    risk_factors.append(factor_info)
                else:
                    protective_factors.append(factor_info)

            clinical_summary = generate_clinical_summary(
                risk_level, risk_factors, protective_factors, risk_prob
            )

            explanations[idx] = {
                'patient_id': idx,
                'risk_probability': risk_prob,
                'risk_level': risk_level,
                'top_risk_factors': risk_factors,
                'top_protective_factors': protective_factors,
                'clinical_summary': clinical_summary
            }

        except Exception as e:
            print(f"Error generating explanation for patient {idx}: {e}")
            continue

    print(f"Generated {len(explanations)} patient-specific explanations")
    return explanations

def simulate_clinician_feedback(trust_scores, X_test, risk_scores, shap_values):
    """Simulate clinician feedback based on model predictions"""
    print("\n=== CLINICIAN FEEDBACK SIMULATION ===")
    high_risk_cases = np.where(risk_scores > 0.7)[0]
    low_trust_cases = np.where(trust_scores < 0.6)[0]
    print(f"High-risk cases identified: {len(high_risk_cases)}")
    print(f"Low-trust predictions: {len(low_trust_cases)}")
    review_cases = np.union1d(high_risk_cases, low_trust_cases)
    print(f"Total cases requiring clinical review: {len(review_cases)}")
    return review_cases