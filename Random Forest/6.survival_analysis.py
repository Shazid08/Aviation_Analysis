import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import joblib


class SurvivalProbabilityAnalyzer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.features = ['year', 'aboard', 'ground', 'fatalities_ratio', 'ground_to_aboard_ratio']
        self.target = 'survival_rate'
        self.scaler = StandardScaler()
        self.model = None
        self.bootstrap_predictions = None

    def prepare_data(self):
        """Prepare data for modeling"""
        X = self.data[self.features]
        y = self.data[self.target]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.features)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        return self

    def train_probability_model(self):
        """Train ensemble model for probability estimation"""
        # Train Random Forest with probability estimation
        self.model = RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=5,
            random_state=42,
            bootstrap=True
        )
        self.model.fit(self.X_train, self.y_train)

        # Generate bootstrap predictions for uncertainty estimation
        self.bootstrap_predictions = np.zeros((len(self.X_test), 100))
        for i in range(100):
            # Bootstrap sample
            idx = np.random.randint(0, len(self.X_train), len(self.X_train))
            X_boot = self.X_train.iloc[idx]
            y_boot = self.y_train.iloc[idx]

            # Train and predict
            model_boot = RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                random_state=i
            )
            model_boot.fit(X_boot, y_boot)
            self.bootstrap_predictions[:, i] = model_boot.predict(self.X_test)

        return self

    def calculate_survival_probabilities(self):
        """Calculate survival probabilities with confidence intervals"""
        # Mean predictions
        y_pred_mean = self.model.predict(self.X_test)

        # Calculate confidence intervals from bootstrap predictions
        y_pred_lower = np.percentile(self.bootstrap_predictions, 2.5, axis=1)
        y_pred_upper = np.percentile(self.bootstrap_predictions, 97.5, axis=1)

        return pd.DataFrame({
            'actual': self.y_test,
            'predicted': y_pred_mean,
            'lower_bound': y_pred_lower,
            'upper_bound': y_pred_upper
        })

    def analyze_risk_factors(self):
        """Analyze key risk factors affecting survival probability"""
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calculate partial dependence for top features
        top_features = feature_importance.head(3)['feature'].values
        partial_dependence = {}

        for feature in top_features:
            # Create feature grid
            feature_idx = self.features.index(feature)
            feature_values = np.linspace(
                self.X_test.iloc[:, feature_idx].min(),
                self.X_test.iloc[:, feature_idx].max(),
                50
            )

            # Calculate partial dependence
            pd_values = []
            for value in feature_values:
                X_temp = self.X_test.copy()
                X_temp.iloc[:, feature_idx] = value
                pd_values.append(self.model.predict(X_temp).mean())

            partial_dependence[feature] = {
                'values': feature_values,
                'effects': pd_values
            }

        return feature_importance, partial_dependence

    def plot_probability_analysis(self, probabilities, risk_factors):
        """Create comprehensive probability analysis plots"""
        plt.figure(figsize=(20, 12))

        # 1. Predicted vs Actual with confidence intervals
        ax1 = plt.subplot(231)
        plt.scatter(probabilities['actual'], probabilities['predicted'], alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.fill_between(
            probabilities['actual'],
            probabilities['lower_bound'],
            probabilities['upper_bound'],
            alpha=0.2
        )
        plt.title('Predicted vs Actual Survival Probabilities')
        plt.xlabel('Actual Survival Rate')
        plt.ylabel('Predicted Survival Rate')

        # 2. Feature Importance
        ax2 = plt.subplot(232)
        feature_importance = risk_factors[0]
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance for Survival Probability')

        # 3. Partial Dependence Plots
        partial_dependence = risk_factors[1]
        for i, (feature, values) in enumerate(partial_dependence.items(), 1):
            ax = plt.subplot(2, 3, i + 3)
            plt.plot(values['values'], values['effects'])
            plt.title(f'Partial Dependence Plot: {feature}')
            plt.xlabel(feature)
            plt.ylabel('Survival Probability')

        plt.tight_layout()
        plt.show()

    def calculate_risk_scores(self):
        """Calculate risk scores for different scenarios"""
        # Create representative scenarios
        scenarios = pd.DataFrame({
            'year': [2020, 2020, 2020],
            'aboard': [100, 50, 200],
            'ground': [0, 0, 0],
            'fatalities_ratio': [0.2, 0.5, 0.8],
            'ground_to_aboard_ratio': [0, 0, 0]
        }, index=['Low Risk', 'Medium Risk', 'High Risk'])

        # Scale scenarios
        scenarios_scaled = pd.DataFrame(
            self.scaler.transform(scenarios),
            columns=self.features,
            index=scenarios.index
        )

        # Predict probabilities
        predictions = []
        for _, scenario in scenarios_scaled.iterrows():
            scenario_reshaped = scenario.values.reshape(1, -1)
            pred = self.model.predict(scenario_reshaped)[0]

            # Bootstrap predictions for confidence interval
            boot_preds = []
            for estimator in self.model.estimators_:
                boot_preds.append(estimator.predict(scenario_reshaped)[0])

            ci_lower = np.percentile(boot_preds, 2.5)
            ci_upper = np.percentile(boot_preds, 97.5)

            predictions.append({
                'mean': pred,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

        return pd.DataFrame(predictions, index=scenarios.index)

    def print_probability_summary(self, risk_scores):
        """Print comprehensive probability analysis summary"""
        print("\n" + "=" * 60)
        print(" Survival Probability Analysis ".center(60, "="))
        print("=" * 60 + "\n")

        print("Model Performance:")
        y_pred = self.model.predict(self.X_test)
        print(f"  R² Score:             {r2_score(self.y_test, y_pred):.4f}")
        print(f"  Mean Absolute Error:  {mean_absolute_error(self.y_test, y_pred):.4f}")
        print(f"  Root Mean Sq Error:   {np.sqrt(mean_squared_error(self.y_test, y_pred)):.4f}")

        print("\nRisk Scenarios (Survival Probability):")
        for scenario in risk_scores.index:
            print(f"\n  {scenario}:")
            print(f"    Mean Probability:    {risk_scores.loc[scenario, 'mean']:.4f}")
            print(f"    95% CI:             ({risk_scores.loc[scenario, 'ci_lower']:.4f}, "
                  f"{risk_scores.loc[scenario, 'ci_upper']:.4f})")


# Run analysis
if __name__ == "__main__":
    try:
        analyzer = SurvivalProbabilityAnalyzer('Enhanced_Airplane_Crashes.csv')

        # Train and evaluate model
        analyzer.prepare_data()
        analyzer.train_probability_model()

        # Calculate probabilities and risk factors
        probabilities = analyzer.calculate_survival_probabilities()
        risk_factors = analyzer.analyze_risk_factors()

        # Generate visualizations
        analyzer.plot_probability_analysis(probabilities, risk_factors)

        # Calculate and print risk scores
        risk_scores = analyzer.calculate_risk_scores()
        analyzer.print_probability_summary(risk_scores)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()