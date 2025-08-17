import pandas as pd
import numpy as np
import pickle
import json
import warnings
import shap
import datetime
warnings.filterwarnings('ignore')

class StrokeRiskPredictor:
    def __init__(self, model_path='/content/drive/MyDrive/SAVED MODEL/xgboost_model.pkl'):
        """Initialize the stroke risk predictor"""
        self.model = None
        self.feature_names = None
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = pickle.load(open(model_path, 'rb'))
            self.feature_names = [
                'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                'gender_encoded', 'ever_married_encoded', 'work_type_encoded',
                'residence_encoded', 'smoking_status_encoded'
            ]
            print("✅ Model loaded successfully!")
            print(f"Expected features: {self.feature_names}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure the model file exists at the specified path")

    def get_patient_input(self):
        """Interactive function to get patient data from user"""
        print("\n" + "="*60)
        print("🏥 MANUAL PATIENT DATA ENTRY")
        print("="*60)

        patient_data = {}
        while True:
            try:
                age = float(input("Enter patient age (0-120): "))
                if 0 <= age <= 120:
                    patient_data['age'] = age
                    break
                else:
                    print("Please enter a valid age between 0 and 120")
            except ValueError:
                print("Please enter a valid number")

        while True:
            hypertension = input("Does patient have hypertension? (yes/no): ").lower().strip()
            if hypertension in ['yes', 'y', '1']:
                patient_data['hypertension'] = 1
                break
            elif hypertension in ['no', 'n', '0']:
                patient_data['hypertension'] = 0
                break
            else:
                print("Please enter 'yes' or 'no'")

        while True:
            heart_disease = input("Does patient have heart disease? (yes/no): ").lower().strip()
            if heart_disease in ['yes', 'y', '1']:
                patient_data['heart_disease'] = 1
                break
            elif heart_disease in ['no', 'n', '0']:
                patient_data['heart_disease'] = 0
                break
            else:
                print("Please enter 'yes' or 'no'")

        while True:
            try:
                glucose = float(input("Enter average glucose level (mg/dL) (50-300): "))
                if 50 <= glucose <= 300:
                    patient_data['avg_glucose_level'] = glucose
                    break
                else:
                    print("Please enter a glucose level between 50 and 300")
            except ValueError:
                print("Please enter a valid number")

        while True:
            try:
                bmi = float(input("Enter BMI (10-60): "))
                if 10 <= bmi <= 60:
                    patient_data['bmi'] = bmi
                    break
                else:
                    print("Please enter a BMI between 10 and 60")
            except ValueError:
                print("Please enter a valid number")

        while True:
            gender = input("Enter gender (Male/Female/Other): ").lower().strip()
            if gender in ['male', 'm']:
                patient_data['gender_encoded'] = 1
                break
            elif gender in ['female', 'f']:
                patient_data['gender_encoded'] = 0
                break
            elif gender in ['other', 'o']:
                patient_data['gender_encoded'] = 0.5
                break
            else:
                print("Please enter 'Male', 'Female', or 'Other'")

        while True:
            married = input("Is patient ever married? (yes/no): ").lower().strip()
            if married in ['yes', 'y', '1']:
                patient_data['ever_married_encoded'] = 1
                break
            elif married in ['no', 'n', '0']:
                patient_data['ever_married_encoded'] = 0
                break
            else:
                print("Please enter 'yes' or 'no'")

        print("\nWork Type Options:")
        print("1. Private")
        print("2. Self-employed")
        print("3. Government job")
        print("4. Children/Never worked")
        while True:
            try:
                work_choice = int(input("Select work type (1-4): "))
                if work_choice == 1:
                    patient_data['work_type_encoded'] = 3
                    break
                elif work_choice == 2:
                    patient_data['work_type_encoded'] = 2
                    break
                elif work_choice == 3:
                    patient_data['work_type_encoded'] = 1
                    break
                elif work_choice == 4:
                    patient_data['work_type_encoded'] = 0
                    break
                else:
                    print("Please enter a number between 1 and 4")
            except ValueError:
                print("Please enter a valid number")

        while True:
            residence = input("Residence type (Urban/Rural): ").lower().strip()
            if residence in ['urban', 'u']:
                patient_data['residence_encoded'] = 1
                break
            elif residence in ['rural', 'r']:
                patient_data['residence_encoded'] = 0
                break
            else:
                print("Please enter 'Urban' or 'Rural'")

        print("\nSmoking Status Options:")
        print("1. Never smoked")
        print("2. Formerly smoked")
        print("3. Currently smokes")
        print("4. Unknown")
        while True:
            try:
                smoking_choice = int(input("Select smoking status (1-4): "))
                if smoking_choice == 1:
                    patient_data['smoking_status_encoded'] = 1
                    break
                elif smoking_choice == 2:
                    patient_data['smoking_status_encoded'] = 2
                    break
                elif smoking_choice == 3:
                    patient_data['smoking_status_encoded'] = 3
                    break
                elif smoking_choice == 4:
                    patient_data['smoking_status_encoded'] = 0
                    break
                else:
                    print("Please enter a number between 1 and 4")
            except ValueError:
                print("Please enter a valid number")

        return patient_data

    def preprocess_patient_data(self, patient_data):
        """Preprocess patient data to match training format"""
        patient_df = pd.DataFrame([patient_data])
        for feature in self.feature_names:
            if feature not in patient_df.columns:
                patient_df[feature] = 0
        patient_df = patient_df[self.feature_names]
        return patient_df

    def predict_risk(self, patient_data):
        """Predict stroke risk for a patient"""
        if self.model is None:
            print("❌ Model not loaded!")
            return None

        try:
            patient_df = self.preprocess_patient_data(patient_data)
            risk_probability = self.model.predict_proba(patient_df)[0, 1]
            prediction = self.model.predict(patient_df)[0]
            if risk_probability > 0.7:
                risk_level = "HIGH"
            elif risk_probability > 0.3:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"

            return {
                'risk_probability': risk_probability,
                'prediction': prediction,
                'risk_level': risk_level,
                'patient_data': patient_data
            }
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None

    def get_shap_explanation(self, patient_data):
        """Get SHAP explanation for the patient"""
        try:
            patient_df = self.preprocess_patient_data(patient_data)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(patient_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            return shap_values[0]
        except Exception as e:
            print(f"Warning: Could not compute SHAP explanations: {e}")
            return None

    def interpret_feature_contribution(self, feature_name, feature_value, contribution):
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

        factor_type = 'risk' if contribution > 0 else 'protective'
        if feature_name in interpretations:
            return interpretations[feature_name][factor_type]
        else:
            if factor_type == 'risk':
                return f"{feature_name} (value: {feature_value:.2f}) contributes to increased stroke risk"
            else:
                return f"{feature_name} (value: {feature_value:.2f}) provides protection against stroke"

    def generate_clinical_summary(self, prediction_result, shap_values=None):
        """Generate clinical summary without recommendations"""
        risk_probability = prediction_result['risk_probability']
        risk_level = prediction_result['risk_level']
        patient_data = prediction_result['patient_data']

        summary = f"\n{'='*60}\n"
        summary += f"🏥 STROKE RISK ASSESSMENT REPORT\n"
        summary += f"{'='*60}\n\n"

        summary += f"📊 RISK OVERVIEW:\n"
        summary += f"   Risk Probability: {risk_probability:.1%}\n"
        summary += f"   Risk Level: {risk_level}\n"
        summary += f"   Prediction: {'HIGH RISK' if prediction_result['prediction'] == 1 else 'LOW RISK'}\n\n"

        summary += f"👤 PATIENT INFORMATION:\n"
        summary += f"   Age: {patient_data['age']:.0f} years\n"
        summary += f"   Gender: {'Male' if patient_data['gender_encoded'] == 1 else 'Female' if patient_data['gender_encoded'] == 0 else 'Other'}\n"
        summary += f"   BMI: {patient_data['bmi']:.1f}\n"
        summary += f"   Glucose Level: {patient_data['avg_glucose_level']:.1f} mg/dL\n"
        summary += f"   Hypertension: {'Yes' if patient_data['hypertension'] == 1 else 'No'}\n"
        summary += f"   Heart Disease: {'Yes' if patient_data['heart_disease'] == 1 else 'No'}\n\n"

        if shap_values is not None:
            feature_contributions = list(zip(self.feature_names, shap_values, [patient_data[f] for f in self.feature_names]))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            risk_factors = [(f, c, v) for f, c, v in feature_contributions if c > 0][:3]
            protective_factors = [(f, c, v) for f, c, v in feature_contributions if c < 0][:3]

            if risk_factors:
                summary += f"🔴 PRIMARY RISK FACTORS:\n"
                for i, (feature, contribution, value) in enumerate(risk_factors, 1):
                    interpretation = self.interpret_feature_contribution(feature, value, contribution)
                    summary += f"   {i}. {interpretation}\n"
                summary += "\n"

            if protective_factors:
                summary += f"🟢 PROTECTIVE FACTORS:\n"
                for i, (feature, contribution, value) in enumerate(protective_factors, 1):
                    interpretation = self.interpret_feature_contribution(feature, value, contribution)
                    summary += f"   {i}. {interpretation}\n"
                summary += "\n"

        summary += f"{'='*60}\n"
        return summary

    def assess_patient(self):
        """Complete patient assessment workflow"""
        print("🏥 Starting Manual Patient Assessment...")
        patient_data = self.get_patient_input()
        print("\n🔄 Processing prediction...")
        prediction_result = self.predict_risk(patient_data)
        if prediction_result is None:
            print("❌ Failed to generate prediction")
            return
        print("🧠 Generating explanations...")
        shap_values = self.get_shap_explanation(patient_data)
        clinical_summary = self.generate_clinical_summary(prediction_result, shap_values)
        print(clinical_summary)
        save_results = input("\n💾 Would you like to save these results? (yes/no): ").lower().strip()
        if save_results in ['yes', 'y', '1']:
            self.save_patient_results(prediction_result, clinical_summary, shap_values)
        return prediction_result, clinical_summary

    def save_patient_results(self, prediction_result, clinical_summary, shap_values):
        """Save patient assessment results"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_data = {
                'timestamp': timestamp,
                'patient_data': prediction_result['patient_data'],
                'risk_probability': float(prediction_result['risk_probability']),
                'risk_level': prediction_result['risk_level'],
                'prediction': int(prediction_result['prediction']),
                'clinical_summary': clinical_summary
            }
            if shap_values is not None:
                results_data['shap_contributions'] = {
                    feature: float(contrib) for feature, contrib in zip(self.feature_names, shap_values)
                }
            filename = f"/content/drive/MyDrive/SAVED MODEL/patient_assessment_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            summary_filename = f"/content/drive/MyDrive/SAVED MODEL/patient_summary_{timestamp}.txt"
            with open(summary_filename, 'w') as f:
                f.write(clinical_summary)
            print(f"✅ Results saved to:")
            print(f"   📄 {filename}")
            print(f"   📝 {summary_filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def quick_assess_patient(age, hypertension, heart_disease, glucose, bmi, gender, married, work_type, residence, smoking):
    """Quick assessment function with predefined values"""
    predictor = StrokeRiskPredictor()
    patient_data = {
        'age': float(age),
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'avg_glucose_level': float(glucose),
        'bmi': float(bmi),
        'gender_encoded': 1 if gender.lower() == 'male' else 0 if gender.lower() == 'female' else 0.5,
        'ever_married_encoded': int(married),
        'work_type_encoded': int(work_type),
        'residence_encoded': 1 if residence.lower() == 'urban' else 0,
        'smoking_status_encoded': int(smoking)
    }
    prediction_result = predictor.predict_risk(patient_data)
    shap_values = predictor.get_shap_explanation(patient_data)
    clinical_summary = predictor.generate_clinical_summary(prediction_result, shap_values)
    print(clinical_summary)
    return prediction_result

def main():
    """Main function to run the patient assessment"""
    print("🏥 STROKE RISK ASSESSMENT SYSTEM")
    print("="*60)
    while True:
        print("\nOptions:")
        print("1. Manual Patient Assessment (Interactive)")
        print("2. Quick Assessment (Pre-filled Example)")
        print("3. Exit")
        choice = input("\nSelect option (1-3): ").strip()
        if choice == '1':
            predictor = StrokeRiskPredictor()
            if predictor.model is not None:
                predictor.assess_patient()
            else:
                print("❌ Model not available. Please check the model path.")
        elif choice == '2':
            print("\n📋 Running example assessment...")
            quick_assess_patient(
                age=67, hypertension=1, heart_disease=0, glucose=228.69, bmi=36.6,
                gender='male', married=1, work_type=3, residence='urban', smoking=2
            )
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == '__main__':
    main()