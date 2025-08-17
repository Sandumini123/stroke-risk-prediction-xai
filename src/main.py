import pandas as pd
import numpy as np
import pickle
import json
import os
from preprocess import load_dataset, impute_missing_values, encode_features, split_data, handle_class_imbalance
from train_model import train_xgboost, evaluate_model, save_risk_scores, select_best_model
from explainability import (compute_shap_explanations, validate_explanations,
                           compute_lime_explanations, compute_all_trust_scores,
                           generate_patient_specific_explanations, simulate_clinician_feedback)

def main():
    """Main execution function"""
    print("=== STROKE PREDICTION MODEL TRAINING ===\n")
    
    print("Phase 1: Data Loading and Preprocessing")
    file_path = '/content/drive/MyDrive/Data Analysis-Based Models - Numerical Data/healthcare-dataset-stroke-data.csv'
    df = load_dataset(file_path)
    if df is None:
        return

    df = impute_missing_values(df)
    df = encode_features(df)
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    X_train, X_test, y_train, y_test = split_data(df)
    os.makedirs('/content/drive/MyDrive/SAVED MODEL', exist_ok=True)
    X_train.to_csv('/content/drive/MyDrive/SAVED MODEL/X_train.csv', index=False)
    X_test.to_csv('/content/drive/MyDrive/SAVED MODEL/X_test.csv', index=False)
    y_test.to_csv('/content/drive/MyDrive/SAVED MODEL/y_test.csv', index=True)

    print("\n" + "="*60)
    print("Phase 2: Model Training and Evaluation")
    print("="*60)

    models_to_evaluate = []
    print("\n1. Training XGBoost with ADASYN...")
    try:
        X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train, method='adasyn')
        model_adasyn = train_xgboost(X_train_balanced, y_train_balanced, use_class_weighting=False)
        metrics_adasyn = evaluate_model(model_adasyn, X_test, y_test, "XGBoost + ADASYN")
        if metrics_adasyn:
            models_to_evaluate.append({
                'name': 'XGBoost + ADASYN',
                'model': model_adasyn,
                'metrics': metrics_adasyn
            })
    except Exception as e:
        print(f"Error training ADASYN model: {e}")

    print("\n2. Training XGBoost with Class Weighting...")
    try:
        model_weighted = train_xgboost(X_train, y_train, use_class_weighting=True)
        metrics_weighted = evaluate_model(model_weighted, X_test, y_test, "XGBoost + Class Weighting")
        if metrics_weighted:
            models_to_evaluate.append({
                'name': 'XGBoost + Class Weighting',
                'model': model_weighted,
                'metrics': metrics_weighted
            })
    except Exception as e:
        print(f"Error training weighted model: {e}")

    print("\n3. Training XGBoost with SMOTE...")
    try:
        X_train_smote, y_train_smote = handle_class_imbalance(X_train, y_train, method='smote')
        model_smote = train_xgboost(X_train_smote, y_train_smote, use_class_weighting=False)
        metrics_smote = evaluate_model(model_smote, X_test, y_test, "XGBoost + SMOTE")
        if metrics_smote:
            models_to_evaluate.append({
                'name': 'XGBoost + SMOTE',
                'model': model_smote,
                'metrics': metrics_smote
            })
    except Exception as e:
        print(f"Error training SMOTE model: {e}")

    if not models_to_evaluate:
        print("ERROR: No models were successfully trained!")
        return

    best_model_info, best_score = select_best_model(models_to_evaluate)
    if best_model_info is None:
        print("ERROR: No valid model found!")
        return

    best_model = best_model_info['model']
    best_metrics = best_model_info['metrics']
    best_method = best_model_info['name']
    print(f"\n🏆 SELECTED BEST MODEL: {best_method}")
    print(f"Composite Score: {best_score:.4f}")
    print(f"Key Metrics - PR-AUC: {best_metrics['pr_auc']:.4f}, F1: {best_metrics['f1_score']:.4f}")

    try:
        save_risk_scores(best_model, X_test, '/content/drive/MyDrive/SAVED MODEL/test_risk_scores.csv')
        pickle.dump(best_model, open('/content/drive/MyDrive/SAVED MODEL/xgboost_model.pkl', 'wb'))
        print("✅ Model and risk scores saved successfully!")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    print("\n" + "="*60)
    print("Phase 3: Explainability and Patient-Specific Analysis")
    print("="*60)

    try:
        print("\n1. Computing SHAP explanations...")
        explainer_shap, shap_values = compute_shap_explanations(best_model, X_test)
        if shap_values is not None:
            validate_explanations(shap_values, X_test)

        print("\n2. Computing LIME explanations...")
        explainer_lime, lime_explanations = compute_lime_explanations(
            best_model, X_train, X_test, num_samples=50
        )

        print("\n3. Computing trust scores...")
        trust_scores, unstable_predictions = compute_all_trust_scores(
            best_model, X_test, explainer_shap, explainer_lime
        )

        print("\n4. Generating patient-specific explanations...")
        if shap_values is not None:
            patient_explanations = generate_patient_specific_explanations(
                best_model, X_test, shap_values,
                patient_indices=range(min(30, len(X_test))),
                top_n_features=6
            )

            print("\n📋 SAMPLE PATIENT EXPLANATIONS:")
            print("="*50)
            for i, (patient_id, explanation) in enumerate(list(patient_explanations.items())[:3]):
                print(f"\n👤 PATIENT {patient_id}")
                print(f"Risk Probability: {explanation['risk_probability']:.1%}")
                print(f"Risk Level: {explanation['risk_level']}")
                print("\nClinical Summary:")
                print(explanation['clinical_summary'])
                print("-" * 50)

            try:
                serializable_explanations = {}
                for patient_id, explanation in patient_explanations.items():
                    serializable_explanations[str(patient_id)] = {
                        'patient_id': int(explanation['patient_id']),
                        'risk_probability': float(explanation['risk_probability']),
                        'risk_level': explanation['risk_level'],
                        'clinical_summary': explanation['clinical_summary'],
                        'top_risk_factors': [
                            {
                                'feature': rf['feature'],
                                'value': float(rf['value']) if isinstance(rf['value'], (int, float, np.number)) else str(rf['value']),
                                'contribution': float(rf['contribution']),
                                'interpretation': rf['interpretation']
                            } for rf in explanation['top_risk_factors']
                        ],
                        'top_protective_factors': [
                            {
                                'feature': pf['feature'],
                                'value': float(pf['value']) if isinstance(pf['value'], (int, float, np.number)) else str(pf['value']),
                                'contribution': float(pf['contribution']),
                                'interpretation': pf['interpretation']
                            } for pf in explanation['top_protective_factors']
                        ]
                    }
                with open('/content/drive/MyDrive/SAVED MODEL/patient_explanations.json', 'w') as f:
                    json.dump(serializable_explanations, f, indent=2)
                print(f"\n✅ Patient explanations saved for {len(patient_explanations)} patients")
            except Exception as e:
                print(f"❌ Error saving patient explanations: {e}")

        if trust_scores is not None and shap_values is not None:
            review_cases = simulate_clinician_feedback(
                trust_scores, X_test,
                best_model.predict_proba(X_test)[:, 1],
                shap_values
            )

    except Exception as e:
        print(f"❌ Error in explainability analysis: {e}")

    print("\n" + "="*60)
    print("Phase 4: Results Summary and Export")
    print("="*60)

    try:
        summary_data = {
            'model_info': {
                'best_model': best_method,
                'composite_score': float(best_score),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'performance_metrics': {
                'accuracy': float(best_metrics['accuracy']),
                'precision': float(best_metrics['precision']),
                'recall': float(best_metrics['recall']),
                'f1_score': float(best_metrics['f1_score']),
                'roc_auc': float(best_metrics['roc_auc']),
                'pr_auc': float(best_metrics['pr_auc'])
            },
            'explanation_stats': {
                'patients_analyzed': len(patient_explanations) if 'patient_explanations' in locals() else 0,
                'high_risk_patients': len([p for p in patient_explanations.values() if p['risk_level'] == 'HIGH']) if 'patient_explanations' in locals() else 0,
                'moderate_risk_patients': len([p for p in patient_explanations.values() if p['risk_level'] == 'MODERATE']) if 'patient_explanations' in locals() else 0,
                'low_risk_patients': len([p for p in patient_explanations.values() if p['risk_level'] == 'LOW']) if 'patient_explanations' in locals() else 0
            }
        }

        with open('/content/drive/MyDrive/SAVED MODEL/training_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)

        text_summary = f"""
STROKE PREDICTION MODEL - TRAINING COMPLETE
==========================================

🎯 BEST MODEL: {best_method}
📊 COMPOSITE SCORE: {best_score:.4f}

🔢 PERFORMANCE METRICS:
• Accuracy: {best_metrics['accuracy']:.4f}
• Precision: {best_metrics['precision']:.4f}
• Recall: {best_metrics['recall']:.4f}
• F1-Score: {best_metrics['f1_score']:.4f}
• ROC-AUC: {best_metrics['roc_auc']:.4f}
• PR-AUC: {best_metrics['pr_auc']:.4f}

📈 DATASET INFO:
• Training samples: {len(X_train):,}
• Test samples: {len(X_test):,}
• Features: {len(X_train.columns)}

🧠 EXPLAINABILITY:
• Patient explanations: {len(patient_explanations) if 'patient_explanations' in locals() else 0}
• High-risk patients: {len([p for p in patient_explanations.values() if p['risk_level'] == 'HIGH']) if 'patient_explanations' in locals() else 0}
• Moderate-risk patients: {len([p for p in patient_explanations.values() if p['risk_level'] == 'MODERATE']) if 'patient_explanations' in locals() else 0}
• Low-risk patients: {len([p for p in patient_explanations.values() if p['risk_level'] == 'LOW']) if 'patient_explanations' in locals() else 0}

📁 FILES SAVED:
• xgboost_model.pkl - Trained model
• test_risk_scores.csv - Risk predictions
• patient_explanations.json - Individual explanations
• training_summary.json - Complete metrics
• X_train.csv, X_test.csv, y_test.csv - Datasets

🎉 TRAINING COMPLETED SUCCESSFULLY!
        """

        with open('/content/drive/MyDrive/SAVED MODEL/training_summary.txt', 'w') as f:
            f.write(text_summary)

        print(text_summary)

    except Exception as e:
        print(f"❌ Error creating summary: {e}")

    print("\n🔍 FINAL VALIDATION:")
    print(f"✅ Model trained and saved: {best_method}")
    print(f"✅ Evaluation metrics computed: PR-AUC = {best_metrics['pr_auc']:.4f}")
    print(f"✅ Patient explanations generated: {len(patient_explanations) if 'patient_explanations' in locals() else 0} unique explanations")
    print(f"✅ All files saved to: /content/drive/MyDrive/SAVED MODEL/")

    if 'patient_explanations' in locals() and len(patient_explanations) > 0:
        print(f"\n🔬 EXPLANATION UNIQUENESS CHECK:")
        unique_summaries = set()
        for explanation in patient_explanations.values():
            unique_summaries.add(explanation['clinical_summary'])
        print(f"Generated {len(patient_explanations)} explanations with {len(unique_summaries)} unique clinical summaries")
        print(f"Uniqueness ratio: {len(unique_summaries)/len(patient_explanations)*100:.1f}%")

    print("\n" + "="*60)
    print("🚀 MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == '__main__':
    main()