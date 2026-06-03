import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier # Requires: pip install xgboost
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

class LiftingPredictor:
    def __init__(self, csv_path):
        self.features = ['Age', 'Body Weight in Pounds', 'Height in Inches', 
                         'How many years have you been lifting?', 'BMI']
        self.df = pd.read_csv(csv_path)
        self.prepare_base_data()
        self.model = None
        self.explainer = None
        self.lift_type = None
        self.target_weight = None

    def prepare_base_data(self):
        """Calculates BMI and cleans data types."""
        bw_kg = self.df['Body Weight in Pounds'] * 0.453592
        h_m = self.df['Height in Inches'] * 0.0254
        self.df['BMI'] = bw_kg / (h_m ** 2)
        
        # Ensure lifting columns are numeric
        cols = ['Back Squat 1RM in Pounds', 'Bench 1RM in Pounds', 'Deadlift 1RM in Pounds']
        for col in cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Calculate Total SBD
        self.df['Total SBD'] = self.df[cols].sum(axis=1, min_count=1)

    def train_for_lift(self, lift_type, target_weight):
        """
        Trains XGBClassifier by EXCLUDING missing values 
        instead of filling them with a median.
        """
        self.lift_type = lift_type
        self.target_weight = target_weight
        
        # 1. DROP MISSING VALUES: Filter the dataset to only include rows where 
        # both the target lift and ALL features are present.
        required_cols = self.features + [lift_type]
        clean_df = self.df.dropna(subset=required_cols)
        
        y = (clean_df[lift_type] >= target_weight).astype(int)
        X = clean_df[self.features]

        # Train XGBClassifier
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.model.fit(X, y)

        # Setup LIME Explainer using the cleaned training data
        self.explainer = LimeTabularExplainer(
            training_data=np.array(X),
            feature_names=self.features,
            class_names=[f'Under {target_weight}', f'{target_weight}+'],
            mode='classification'
        )
        
        print(f"--- Model trained for {target_weight}lb+ {lift_type} (XGBoost) ---")
        print(f"Dataset Size: {len(clean_df)} complete lifter profiles used (NaNs excluded).")

    def predict_probability(self, weight, age=None, height_inches=None, years_lifting=None, show_lime=True):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_for_lift first.")

        # Check if any required data is missing
        if any(v is None for v in [weight, age, height_inches, years_lifting]):
            print("\nError: This model was trained on complete profiles only.")
            print("Please provide all values (Weight, Age, Height, Years Lifting).")
            return

        # Calculate BMI
        bmi = (weight * 0.453592) / ((height_inches * 0.0254) ** 2)
            
        # Prepare input row
        person_row = pd.DataFrame([[age, weight, height_inches, years_lifting, bmi]], 
                                 columns=self.features)

        # Get Probabilities
        probs = self.model.predict_proba(person_row)[0]
        prob_under, prob_over = probs[0], probs[1]
        
        print("\n" + "="*55)
        print(f"PREDICTION FOR: {self.target_weight}lb+ {self.lift_type}")
        print("-" * 55)
        print(f"Input Stats:")
        print(f"  Weight:        {weight} lbs")
        print(f"  Height:        {height_inches} inches")
        print(f"  Age:           {age}")
        print(f"  Years Lifting: {years_lifting}")
        print(f"  Calculated BMI: {bmi:.2f}")
        print("-" * 55)
        print(f"PROBABILITY OF SUCCESS: {prob_over:.2%}")
        print(f"PROBABILITY OF FAILURE: {prob_under:.2%}")
        print("="*55)
        
        if show_lime:
            exp = self.explainer.explain_instance(
                data_row=person_row.values[0], 
                predict_fn=self.model.predict_proba,
                num_features=len(self.features)
            )
            
            print("\nLIME Influence (Why it gave this probability):")
            for feature, weight_val in exp.as_list():
                impact = "Increases Success" if weight_val > 0 else "Decreases Success"
                print(f" -> {feature:40}: {weight_val:.4f} ({impact})")
            
            exp.as_pyplot_figure()
            plt.title(f"Explanation for {self.target_weight}lb {self.lift_type}")
            plt.show()

# --- RUNNING THE TEST ---
# Update this path to your local file location
csv_path = r'D:\OneDrive\Workout programs\WR 2021 Deadlift Data - Weightroom Survey 2021.csv'
predictor = LiftingPredictor(csv_path)

# Test specifically for a 495lb Deadlift
predictor.train_for_lift('Deadlift 1RM in Pounds', 495)
predictor.predict_probability(weight=158, age=25, height_inches=71, years_lifting=4, show_lime=True)