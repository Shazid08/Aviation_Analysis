import traceback
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import chi2, f_oneway, chisquare, norm
from tabulate import tabulate


class AviationRiskClustering:
    def __init__(self):
        """Initialize the clustering analysis components."""
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = PCA(n_components=0.95)
        self.features = None
        self.selected_features = None
        self.original_columns = None
        self.metrics = None

    def load_and_preprocess_data(self, filepath):
        """
        Load and preprocess the aviation accident data.

        Parameters:
            filepath (str): Path to the data file

        Returns:
            pd.DataFrame: Preprocessed data for analysis
        """
        try:
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=False)
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month

            # Parse time data
            def parse_time(time_str):
                try:
                    if pd.isna(time_str):
                        return np.nan
                    if ':' in str(time_str):
                        return int(str(time_str).split(':')[0])
                    return np.nan
                except:
                    return np.nan

            data['Hour'] = data['Time'].apply(parse_time)
            data['IsMilitary'] = data['Operator'].str.contains('Military', case=False, na=False).astype(int)

            # Convert numeric columns
            numeric_columns = ['Aboard', 'Fatalities', 'Ground']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Calculate derived metrics
            data['Survival_Rate'] = data.apply(
                lambda row: 1 - (row['Fatalities'] / row['Aboard']) if row['Aboard'] > 0 else 0,
                axis=1
            )
            data['Total_Fatalities'] = data['Fatalities'].fillna(0) + data['Ground'].fillna(0)
            data['Fatality_Rate'] = data.apply(
                lambda row: row['Total_Fatalities'] / row['Aboard'] if row['Aboard'] > 0 else 1,
                axis=1
            )
            data['Ground_Impact'] = data.apply(
                lambda row: row['Ground'] / row['Total_Fatalities'] if row['Total_Fatalities'] > 0 else 0,
                axis=1
            )

            # Clip ratios to valid ranges
            data['Fatality_Rate'] = data['Fatality_Rate'].clip(0, 1)
            data['Ground_Impact'] = data['Ground_Impact'].clip(0, 1)
            data['Survival_Rate'] = data['Survival_Rate'].clip(0, 1)

            # Select features for clustering
            self.features = [
                'Year', 'Month', 'Hour', 'IsMilitary',
                'Aboard', 'Total_Fatalities', 'Survival_Rate',
                'Fatality_Rate', 'Ground_Impact'
            ]

            # Handle missing values and outliers
            for feature in self.features:
                data[feature] = data[feature].replace([np.inf, -np.inf], np.nan)
                data[feature] = data[feature].fillna(data[feature].median())

            return data[self.features]

        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            return None

    def find_optimal_k(self, X, k_range=range(2, 11)):
        """
        Determine optimal number of clusters using multiple metrics.
        """
        try:
            results = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)

                result = {
                    'k': k,
                    'inertia': kmeans.inertia_,
                    'silhouette': silhouette_score(X, labels),
                    'calinski': calinski_harabasz_score(X, labels),
                    'davies_bouldin': davies_bouldin_score(X, labels)
                }
                results.append(result)

            results_df = pd.DataFrame(results)
            self.plot_clustering_metrics(results_df)  # Changed from _plot_clustering_metrics
            return results_df

        except Exception as e:
            print(f"Error finding optimal k: {str(e)}")
            return None


    def plot_clustering_metrics(self, results_df):  # Changed from _plot_clustering_metrics
        """
        Visualize clustering performance metrics.
        """
        try:
            plt.figure(figsize=(15, 10))

            # Elbow plot
            plt.subplot(2, 2, 1)
            plt.plot(results_df['k'], results_df['inertia'], 'bx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method')

            # Silhouette score
            plt.subplot(2, 2, 2)
            plt.plot(results_df['k'], results_df['silhouette'], 'rx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Analysis')

            # Calinski-Harabasz score
            plt.subplot(2, 2, 3)
            plt.plot(results_df['k'], results_df['calinski'], 'gx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Calinski-Harabasz Score')
            plt.title('Calinski-Harabasz Analysis')

            # Davies-Bouldin score
            plt.subplot(2, 2, 4)
            plt.plot(results_df['k'], results_df['davies_bouldin'], 'mx-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Davies-Bouldin Score')
            plt.title('Davies-Bouldin Analysis')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting clustering metrics: {str(e)}")

    def train_model(self, X, n_clusters=5):  # Changed from train_kmeans
        """
        Train K-means model and calculate performance metrics.
        """
        try:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = self.kmeans.fit_predict(X)

            # Calculate and store metrics
            self.metrics = {
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels)
            }

            # Add stability score through cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            stability_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
                X_val = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]

                kmeans_temp = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                kmeans_temp.fit(X_train)
                val_labels = kmeans_temp.predict(X_val)
                stability_scores.append(silhouette_score(X_val, val_labels))

            self.metrics['stability_score'] = np.mean(stability_scores)
            self.metrics['stability_std'] = np.std(stability_scores)

            return self.metrics

        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None

    def analyze_clusters(self, X, original_data):
        """
        Analyze clusters and generate profiles.

        Parameters:
            X (np.ndarray): Input data
            original_data (pd.DataFrame): Original dataset

        Returns:
            pd.DataFrame: Cluster profiles
        """
        if self.kmeans is None:
            raise ValueError("Must train model before analyzing clusters")

        try:
            analysis_df = pd.DataFrame({
                'Cluster': self.kmeans.labels_,
                'Month': original_data['Month'],
                'Total_Fatalities': original_data['Total_Fatalities'],
                'Survival_Rate': original_data['Survival_Rate'],
                'IsMilitary': original_data['IsMilitary'],
                'Aboard': original_data['Aboard'],
                'Year': original_data['Year']  # Added Year for temporal analysis
            })

            cluster_profiles = []
            for i in range(self.kmeans.n_clusters):
                cluster_data = analysis_df[analysis_df['Cluster'] == i]

                profile = {
                    'Cluster': i,
                    'Size': len(cluster_data),
                    'Size_Percentage': (len(cluster_data) / len(analysis_df) * 100),
                    'Avg_Fatalities': cluster_data['Total_Fatalities'].mean(),
                    'Max_Fatalities': cluster_data['Total_Fatalities'].max(),
                    'Avg_Survival_Rate': cluster_data['Survival_Rate'].mean(),
                    'Military_Percentage': (cluster_data['IsMilitary'].mean() * 100),
                    'Avg_Aboard': cluster_data['Aboard'].mean(),
                    'Fatal_Accidents': len(cluster_data[cluster_data['Total_Fatalities'] > 0]),
                    'Zero_Survival_Rate': len(cluster_data[cluster_data['Survival_Rate'] == 0]),
                    'Year_Range': f"{cluster_data['Year'].min()}-{cluster_data['Year'].max()}"
                }
                cluster_profiles.append(profile)

            return pd.DataFrame(cluster_profiles)

        except Exception as e:
            print(f"Error in cluster analysis: {str(e)}")
            return None


    def _interpret_cluster(self, cluster_stats):
        """
        Interpret cluster characteristics and generate recommendations.

        Parameters:
            cluster_stats (pd.Series): Statistics for a cluster

        Returns:
            dict: Cluster interpretation and recommendations
        """
        try:
            # Determine size category
            if cluster_stats['Avg_Aboard'] > 40:
                size_category = "Large"
            elif cluster_stats['Avg_Aboard'] > 20:
                size_category = "Medium"
            else:
                size_category = "Small"

            # Determine operation type
            if cluster_stats['Military_Percentage'] > 75:
                operation_type = "Military"
            elif cluster_stats['Military_Percentage'] > 25:
                operation_type = "Mixed Military-Civilian"
            else:
                operation_type = "Civilian"

            # Determine risk level
            survival_rate = cluster_stats['Avg_Survival_Rate']
            fatal_ratio = cluster_stats['Fatal_Accidents'] / cluster_stats['Size']

            if survival_rate < 0.1:
                risk_level = "Critical Risk"
            elif survival_rate < 0.3:
                risk_level = "High Risk"
            elif survival_rate < 0.5:
                risk_level = "Medium Risk"
            else:
                risk_level = "Lower Risk"

            # Generate profile and description
            profile = f"{size_category} {operation_type} Operations"
            description = (
                f"{'High' if survival_rate > 0.5 else 'Low'} survival rate ({survival_rate:.1%}), "
                f"{operation_type.lower()} operations, "
                f"{size_category.lower()} aircraft "
                f"(avg. {cluster_stats['Avg_Aboard']:.1f} passengers)"
            )

            # Generate risk factors and recommendations
            risk_factors = self._assess_detailed_risk_factors(cluster_stats, fatal_ratio)
            recommendations = self._generate_recommendations(cluster_stats, risk_level)

            return {
                'profile': profile,
                'description': description,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendations': recommendations
            }

        except Exception as e:
            print(f"Error interpreting cluster: {str(e)}")
            return None

    def _assess_detailed_risk_factors(self, stats, fatal_ratio):
        """
        Generate detailed risk factors for a cluster.

        Parameters:
            stats (pd.Series): Cluster statistics
            fatal_ratio (float): Ratio of fatal accidents

        Returns:
            str: Semicolon-separated risk factors
        """
        try:
            risk_factors = []

            # Survival-related factors
            if stats['Avg_Survival_Rate'] < 0.1:
                risk_factors.append("Critically low survival rate")
            elif stats['Avg_Survival_Rate'] < 0.3:
                risk_factors.append("Very low survival rate")

            # Operational factors
            if stats['Military_Percentage'] > 75:
                risk_factors.append("Predominantly military operations")
            elif stats['Military_Percentage'] > 25:
                risk_factors.append("Mixed military-civilian operations")

            # Fatality factors
            if fatal_ratio > 0.9:
                risk_factors.append("Extremely high fatal accident ratio")
            elif fatal_ratio > 0.7:
                risk_factors.append("High fatal accident ratio")

            # Passenger load factors
            if stats['Avg_Aboard'] > 40:
                risk_factors.append("High passenger count")

            # Zero survival cases
            zero_survival_ratio = stats['Zero_Survival_Rate'] / stats['Size']
            if zero_survival_ratio > 0.7:
                risk_factors.append("High proportion of zero-survival incidents")

            return "; ".join(risk_factors) if risk_factors else "Standard risk profile"

        except Exception as e:
            print(f"Error assessing risk factors: {str(e)}")
            return "Unable to assess risk factors"


    def _generate_recommendations(self, stats, risk_level):
        """
        Generate specific recommendations based on cluster characteristics.

        Parameters:
            stats (pd.Series): Cluster statistics
            risk_level (str): Risk level category

        Returns:
            list: Prioritized recommendations
        """
        try:
            recommendations = []

            # Risk level recommendations
            if risk_level == "Critical Risk":
                recommendations.extend([
                    "Implement comprehensive survival rate improvement program",
                    "Review and enhance emergency response protocols",
                    "Establish mandatory safety audits"
                ])
            elif risk_level == "High Risk":
                recommendations.extend([
                    "Strengthen existing safety protocols",
                    "Enhanced emergency response training",
                    "Regular safety assessments"
                ])

            # Operation-specific recommendations
            if stats['Military_Percentage'] > 75:
                recommendations.extend([
                    "Develop specialized military safety protocols",
                    "Enhanced combat/training mission risk assessment",
                    "Military-specific emergency procedures"
                ])
            elif stats['Military_Percentage'] > 25:
                recommendations.extend([
                    "Implement dual-standard safety protocols",
                    "Separate risk assessment for military and civilian operations",
                    "Coordinated emergency response procedures"
                ])

            # Aircraft size recommendations
            if stats['Avg_Aboard'] > 40:
                recommendations.extend([
                    "Implement large-aircraft specific safety measures",
                    "Enhanced passenger evacuation procedures",
                    "Regular large-scale emergency drills"
                ])
            elif stats['Avg_Aboard'] < 20:
                recommendations.extend([
                    "Small aircraft safety protocol review",
                    "Enhanced pilot training for small aircraft",
                    "Specific small aircraft emergency procedures"
                ])

            # Survival rate specific
            if stats['Avg_Survival_Rate'] < 0.1:
                recommendations.extend([
                    "Priority focus on survival rate improvement",
                    "Emergency equipment assessment and upgrade",
                    "Enhanced crew survival training"
                ])

            # General recommendations
            recommendations.extend([
                "Regular safety training updates",
                "Enhanced pilot training and risk awareness",
                "Continuous monitoring and assessment"
            ])

            return recommendations

        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations"]

    def generate_report(self, cluster_profiles):
        """
        Generate comprehensive analysis report.

        Parameters:
            cluster_profiles (pd.DataFrame): Cluster profile data

        Returns:
            dict: Comprehensive analysis report
        """
        try:
            # Calculate overall statistics
            total_accidents = cluster_profiles['Size'].sum()
            total_fatal = cluster_profiles['Fatal_Accidents'].sum()
            avg_survival = cluster_profiles['Avg_Survival_Rate'].mean()
            military_pct = cluster_profiles['Military_Percentage'].mean()

            # Calculate standard errors
            survival_stderr = cluster_profiles['Avg_Survival_Rate'].std() / np.sqrt(len(cluster_profiles))
            military_stderr = cluster_profiles['Military_Percentage'].std() / np.sqrt(len(cluster_profiles))

            # Create report structure
            report = {
                'summary_statistics': {
                    'total_accidents': total_accidents,
                    'total_fatal_accidents': total_fatal,
                    'average_survival_rate': avg_survival,
                    'military_accident_percentage': military_pct,
                    'survival_rate_stderr': survival_stderr,
                    'military_stderr': military_stderr
                },
                'cluster_details': {},
                'risk_levels': {},
                'key_findings': [
                    f"Majority of accidents ({cluster_profiles['Size_Percentage'].max():.1f}%) "
                    f"fall in Cluster {cluster_profiles['Size'].idxmax()}",
                    f"Military operations show distinct risk profile with "
                    f"{cluster_profiles['Military_Percentage'].max():.1f}% military presence in "
                    f"Cluster {cluster_profiles['Military_Percentage'].idxmax()}",
                    f"Highest survival rate of {cluster_profiles['Avg_Survival_Rate'].max():.1%} "
                    f"in Cluster {cluster_profiles['Avg_Survival_Rate'].idxmax()}",
                    f"Largest average passenger count of {cluster_profiles['Avg_Aboard'].max():.1f} "
                    f"in Cluster {cluster_profiles['Avg_Aboard'].idxmax()}"
                ]
            }

            # Add cluster details
            for idx, row in cluster_profiles.iterrows():
                # Generate cluster interpretation
                interpretation = self._interpret_cluster(row)

                # Store risk level
                report['risk_levels'][f'Cluster_{idx}'] = interpretation['risk_level']

                # Store detailed cluster information
                report['cluster_details'][f'Cluster_{idx}'] = {
                    'statistics': row.to_dict(),
                    'interpretation': interpretation
                }

            # Add analysis results
            report['analysis'] = {
                'risk_patterns': self._analyze_risk_patterns(cluster_profiles),
                'operation_types': self._analyze_operation_types(cluster_profiles),
                'survival_patterns': self._analyze_survival_patterns(cluster_profiles)
            }

            return report

        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None
    def _calculate_cluster_metrics(self, cluster_stats):
        """
        Calculate additional metrics for cluster analysis.

        Parameters:
            cluster_stats (pd.Series): Statistics for a cluster

        Returns:
            dict: Additional cluster metrics
        """
        try:
            return {
                'fatality_rate': 1 - cluster_stats['Avg_Survival_Rate'],
                'risk_intensity': (1 - cluster_stats['Avg_Survival_Rate']) *
                                  (cluster_stats['Fatal_Accidents'] / cluster_stats['Size']),
                'operation_complexity': 'High' if (cluster_stats['Military_Percentage'] > 25 or
                                                   cluster_stats['Avg_Aboard'] > 40) else 'Moderate'
            }
        except Exception as e:
            print(f"Error calculating cluster metrics: {str(e)}")
            return None


    def _analyze_risk_patterns(self, cluster_profiles):
        """
        Analyze risk patterns across clusters.

        Parameters:
            cluster_profiles (pd.DataFrame): Cluster profile data

        Returns:
            dict: Risk pattern analysis
        """
        try:
            return {
                'high_risk_percentage': len(cluster_profiles[cluster_profiles['Avg_Survival_Rate'] < 0.3]) /
                                        len(cluster_profiles) * 100,
                'critical_clusters': len(cluster_profiles[cluster_profiles['Avg_Survival_Rate'] < 0.1]),
                'safe_clusters': len(cluster_profiles[cluster_profiles['Avg_Survival_Rate'] > 0.5]),
                'risk_trend': 'Concerning' if len(cluster_profiles[cluster_profiles['Avg_Survival_Rate'] < 0.3]) >
                                              len(cluster_profiles) / 2 else 'Moderate'
            }
        except Exception as e:
            print(f"Error analyzing risk patterns: {str(e)}")
            return None

    def _analyze_operation_types(self, cluster_profiles):
        """
        Analyze the distribution of operation types across clusters.

        Parameters:
            cluster_profiles (pd.DataFrame): Cluster profile data

        Returns:
            dict: Operation type analysis
        """
        try:
            return {
                'military_dominant': len(cluster_profiles[cluster_profiles['Military_Percentage'] > 75]),
                'civilian_dominant': len(cluster_profiles[cluster_profiles['Military_Percentage'] < 25]),
                'mixed_operations': len(cluster_profiles[(cluster_profiles['Military_Percentage'] >= 25) &
                                                         (cluster_profiles['Military_Percentage'] <= 75)])
            }
        except Exception as e:
            print(f"Error analyzing operation types: {str(e)}")
            return None


    def _analyze_survival_patterns(self, cluster_profiles):
        """
        Analyze survival patterns across clusters.

        Parameters:
            cluster_profiles (pd.DataFrame): Cluster profile data

        Returns:
            dict: Survival pattern analysis
        """
        try:
            return {
                'average_survival': cluster_profiles['Avg_Survival_Rate'].mean(),
                'survival_range': cluster_profiles['Avg_Survival_Rate'].max() -
                                  cluster_profiles['Avg_Survival_Rate'].min(),
                'clusters_above_average': len(cluster_profiles[
                                                  cluster_profiles['Avg_Survival_Rate'] > cluster_profiles[
                                                      'Avg_Survival_Rate'].mean()
                                                  ]),
                'survival_trend': 'Positive' if cluster_profiles['Avg_Survival_Rate'].mean() > 0.5
                else 'Needs Improvement'
            }
        except Exception as e:
            print(f"Error analyzing survival patterns: {str(e)}")
            return None


    def print_report(self, report):
        """
        Print comprehensive analysis report.

        Parameters:
            report (dict): Analysis report
        """
        if report is None:
            print("Error: No report available to print")
            return

        print("\n" + "=" * 80)
        print("AVIATION ACCIDENT CLUSTERING ANALYSIS REPORT".center(80))
        print("=" * 80)

        sections = [
            ("CLUSTERING PERFORMANCE METRICS", self._print_performance_metrics),
            ("SUMMARY STATISTICS", self._print_summary_stats),
            ("STATISTICAL ANALYSIS", self._print_statistical_tests),
            ("CLUSTER PROFILES", self._print_cluster_profiles),
            ("RISK ANALYSIS", self._print_risk_analysis),
            ("RECOMMENDATIONS", self._print_recommendations)
        ]

        for section_title, section_func in sections:
            print("\n" + "-" * 50)
            print(section_title)
            print("-" * 50)
            try:
                section_func(report)
            except Exception as e:
                print(f"Error printing {section_title.lower()}: {str(e)}")
                print(f"{section_title.lower()} not available")


    def _print_performance_metrics(self, report):
        """Print clustering performance metrics."""
        try:
            silhouette_quality = "poor" if self.metrics['silhouette'] < 0.2 else \
                "fair" if self.metrics['silhouette'] < 0.4 else \
                    "good" if self.metrics['silhouette'] < 0.6 else "excellent"

            stability_quality = "unstable" if self.metrics['stability_score'] < 0.2 else \
                "moderately stable" if self.metrics['stability_score'] < 0.4 else \
                    "stable" if self.metrics['stability_score'] < 0.6 else "highly stable"

            metrics_table = [
                ["Metric", "Value", "Interpretation"],
                ["Silhouette Score",
                 f"{self.metrics['silhouette']:.3f}",
                 f"Clustering quality is {silhouette_quality}"],
                ["Calinski-Harabasz",
                 f"{self.metrics['calinski_harabasz']:.1f}",
                 "Measure of cluster density"],
                ["Stability Score",
                 f"{self.metrics['stability_score']:.3f}",
                 f"Solution is {stability_quality}"],
                ["Stability Std",
                 f"{self.metrics['stability_std']:.3f}",
                 "Variation in stability"]
            ]

            print(tabulate(metrics_table, headers="firstrow", tablefmt="grid"))

        except Exception as e:
            print(f"Error printing performance metrics: {str(e)}")


    def _print_summary_stats(self, report):
        """Print summary statistics with confidence intervals."""
        try:
            stats = report['summary_statistics']
            z = norm.ppf(0.975)  # 95% confidence interval

            summary_table = [
                ["Metric", "Value", "95% Confidence Interval"],
                ["Total Accidents",
                 f"{stats['total_accidents']:,}",
                 "N/A"],
                ["Fatal Accidents",
                 f"{stats['total_fatal_accidents']:,}",
                 f"({stats['total_fatal_accidents'] - z * np.sqrt(stats['total_fatal_accidents']):.0f}, "
                 f"{stats['total_fatal_accidents'] + z * np.sqrt(stats['total_fatal_accidents']):.0f})"],
                ["Average Survival Rate",
                 f"{stats['average_survival_rate']:.1%}",
                 f"({max(0, stats['average_survival_rate'] - z * stats.get('survival_rate_stderr', 0)):.1%}, "
                 f"{min(1, stats['average_survival_rate'] + z * stats.get('survival_rate_stderr', 0)):.1%})"],
                ["Military Accident Percentage",
                 f"{stats['military_accident_percentage']:.1f}%",
                 f"({max(0, stats['military_accident_percentage'] - z * stats.get('military_stderr', 0)):.1f}%, "
                 f"{min(100, stats['military_accident_percentage'] + z * stats.get('military_stderr', 0)):.1f}%)"]
            ]

            print(tabulate(summary_table, headers="firstrow", tablefmt="grid"))

        except Exception as e:
            print(f"Error printing summary stats: {str(e)}")
            print("Summary statistics not available")

    def _print_statistical_tests(self, report):
        """
        Perform and print statistical tests with comprehensive error handling.

        Parameters:
            report (dict): Analysis report
        """
        try:
            # Prepare data for tests
            cluster_details = list(report['cluster_details'].values())
            survival_rates = [details['statistics']['Avg_Survival_Rate'] for details in cluster_details]
            military_percentages = [details['statistics']['Military_Percentage'] for details in cluster_details]

            # ANOVA test with proper array handling
            try:
                f_stat, anova_p = f_oneway(*[[rate] for rate in survival_rates])
                anova_results = {
                    'statistic': f_stat if not np.isnan(f_stat) else "N/A",
                    'p_value': anova_p if not np.isnan(anova_p) else "N/A",
                    'interpretation': "Significant difference" if (not np.isnan(anova_p) and anova_p < 0.05)
                    else "No significant difference"
                }
            except:
                anova_results = {
                    'statistic': "N/A",
                    'p_value': "N/A",
                    'interpretation': "Could not perform ANOVA"
                }

            # Chi-square test for military distribution
            try:
                expected = np.ones_like(military_percentages) * np.mean(military_percentages)
                chi2_stat, chi2_p = chisquare(military_percentages, expected)
                chi2_results = {
                    'statistic': chi2_stat,
                    'p_value': chi2_p,
                    'interpretation': "Non-random distribution" if chi2_p < 0.05 else "Random distribution"
                }
            except:
                chi2_results = {
                    'statistic': "N/A",
                    'p_value': "N/A",
                    'interpretation': "Could not perform chi-square test"
                }

            # Create and print results table
            tests_table = [
                ["Test", "Statistic", "p-value", "Interpretation"],
                ["ANOVA (Survival Rates)",
                 str(anova_results['statistic']),
                 str(anova_results['p_value']),
                 anova_results['interpretation']],
                ["Chi-square (Military)",
                 f"{chi2_results['statistic']:.2f}" if isinstance(chi2_results['statistic'], (int, float))
                 else chi2_results['statistic'],
                 f"{chi2_results['p_value']:.4f}" if isinstance(chi2_results['p_value'], (int, float))
                 else chi2_results['p_value'],
                 chi2_results['interpretation']]
            ]

            print(tabulate(tests_table, headers="firstrow", tablefmt="grid"))

            # Effect size analysis
            if isinstance(anova_results['statistic'], (int, float)) and not np.isnan(anova_results['statistic']):
                n = len(cluster_details)
                eta_squared = anova_results['statistic'] * (n - 1) / \
                              (anova_results['statistic'] * (n - 1) + n * 4)
                print(f"\nEffect size (η²): {eta_squared:.3f}")
                print("Interpretation:",
                      "Small" if eta_squared < 0.06 else
                      "Medium" if eta_squared < 0.14 else
                      "Large")

        except Exception as e:
            print(f"Error in statistical testing: {str(e)}")
            print("Could not complete statistical analysis")


    def _print_cluster_profiles(self, report):
        """
        Print detailed cluster profiles with statistical significance.

        Parameters:
            report (dict): Analysis report
        """
        try:
            profiles = []
            headers = ["Cluster", "Size", "Survival Rate", "Military %", "Risk Level",
                       "Key Characteristic", "Statistical Significance"]

            total_accidents = sum(details['statistics']['Size']
                                  for details in report['cluster_details'].values())

            for cluster, details in report['cluster_details'].items():
                stats = details['statistics']

                # Calculate statistical significance
                expected_size = total_accidents / len(report['cluster_details'])
                chi2_stat = ((stats['Size'] - expected_size) ** 2) / expected_size
                p_value = 1 - chi2.cdf(chi2_stat, df=1)

                profiles.append([
                    cluster,
                    f"{stats['Size']:,} ({stats['Size_Percentage']:.1f}%)",
                    f"{stats['Avg_Survival_Rate']:.1%}",
                    f"{stats['Military_Percentage']:.1f}%",
                    details['interpretation']['risk_level'],
                    self._determine_key_characteristic(stats),
                    "Significant" if p_value < 0.05 else "Not significant"
                ])

            print(tabulate(profiles, headers=headers, tablefmt="grid"))

        except Exception as e:
            print(f"Error printing cluster profiles: {str(e)}")


    def _print_risk_analysis(self, report):
        """
        Print risk analysis with statistical significance.

        Parameters:
            report (dict): Analysis report
        """
        try:
            risk_levels = pd.Series(report['risk_levels'])
            risk_counts = risk_levels.value_counts()
            chi2_stat, chi2_p = stats.chisquare(risk_counts)

            risk_table = [
                ["Risk Level", "Count", "Percentage", "Significant?"],
                *[[level, count, f"{count / len(risk_levels) * 100:.1f}%",
                   "Yes" if chi2_p < 0.05 else "No"]
                  for level, count in risk_counts.items()]
            ]

            print(tabulate(risk_table, headers="firstrow", tablefmt="grid"))

        except Exception as e:
            print(f"Error printing risk analysis: {str(e)}")


    def _print_recommendations(self, report):
        """
        Print prioritized recommendations for each cluster.

        Parameters:
            report (dict): Analysis report
        """
        try:
            for cluster, details in report['cluster_details'].items():
                print(f"\n{cluster}:")
                recommendations = details['interpretation']['recommendations']

                # Prioritize recommendations
                high_priority = recommendations[:3]
                medium_priority = recommendations[3:6]
                low_priority = recommendations[6:]

                # Print high priority recommendations
                print("\nHIGH Priority:")
                for i, rec in enumerate(high_priority, 1):
                    print(f"{i}. {rec}")

                # Print medium priority recommendations
                if medium_priority:
                    print("\nMEDIUM Priority:")
                    for i, rec in enumerate(medium_priority, 1):
                        print(f"{i}. {rec}")

                # Print low priority recommendations
                if low_priority:
                    print("\nLOW Priority:")
                    for i, rec in enumerate(low_priority, 1):
                        print(f"{i}. {rec}")

        except Exception as e:
            print(f"Error printing recommendations: {str(e)}")

    def create_visualizations(self, cluster_profiles, report):
        """
        Create comprehensive visualizations of clustering results.

        Parameters:
            cluster_profiles (pd.DataFrame): Cluster profile data
            report (dict): Analysis report
        """
        try:
            # Set style to a valid matplotlib style
            plt.style.use('default')  # Changed from 'seaborn' to 'default'

            # Create main visualization figure
            fig = plt.figure(figsize=(20, 15))
            gs = GridSpec(3, 3, figure=fig)

            # 1. Cluster Size Distribution
            ax1 = fig.add_subplot(gs[0, 0])
            cluster_sizes = cluster_profiles['Size']
            bars = ax1.bar(range(len(cluster_sizes)), cluster_sizes)
            ax1.set_title('Cluster Size Distribution', fontsize=12, pad=15)
            ax1.set_xlabel('Cluster')
            ax1.set_ylabel('Number of Accidents')
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height):,}',
                         ha='center', va='bottom')

            # 2. Survival Rates Analysis
            ax2 = fig.add_subplot(gs[0, 1])
            survival_data = np.array([
                cluster_profiles['Avg_Survival_Rate'],
                1 - cluster_profiles['Avg_Survival_Rate']
            ])
            im = ax2.imshow(survival_data, aspect='auto', cmap='RdYlGn')
            ax2.set_title('Survival vs Fatality Rates', fontsize=12, pad=15)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Survival Rate', 'Fatality Rate'])
            ax2.set_xticks(range(len(cluster_profiles)))
            ax2.set_xticklabels([f'Cluster {i}' for i in range(len(cluster_profiles))])
            plt.colorbar(im, ax=ax2)

            # 3. Military vs Civilian Distribution
            ax3 = fig.add_subplot(gs[0, 2])
            bottoms = np.zeros(len(cluster_profiles))
            military_pct = cluster_profiles['Military_Percentage']
            civilian_pct = 100 - military_pct
            ax3.bar(range(len(cluster_profiles)), military_pct, label='Military',
                    color='navy', alpha=0.7)
            ax3.bar(range(len(cluster_profiles)), civilian_pct, bottom=military_pct,
                    label='Civilian', color='lightblue', alpha=0.7)
            ax3.set_title('Military vs Civilian Distribution', fontsize=12, pad=15)
            ax3.set_xlabel('Cluster')
            ax3.set_ylabel('Percentage')
            ax3.legend()

            # 4. Risk Assessment Matrix
            ax4 = fig.add_subplot(gs[1, 0:2])
            scatter = ax4.scatter(cluster_profiles['Avg_Aboard'],
                                  cluster_profiles['Avg_Survival_Rate'],
                                  s=cluster_profiles['Size'] / 10,
                                  c=cluster_profiles['Military_Percentage'],
                                  cmap='coolwarm')
            for i, row in cluster_profiles.iterrows():
                ax4.annotate(f'Cluster {i}\n({row["Size"]:,} accidents)',
                             (row['Avg_Aboard'], row['Avg_Survival_Rate']),
                             xytext=(5, 5), textcoords='offset points',
                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            ax4.set_title('Risk Assessment Matrix', fontsize=12, pad=15)
            ax4.set_xlabel('Average Passengers Aboard')
            ax4.set_ylabel('Survival Rate')
            plt.colorbar(scatter, ax=ax4, label='Military Percentage')

            # 5. Risk Level Distribution
            ax5 = fig.add_subplot(gs[1, 2])
            risk_counts = pd.Series(report['risk_levels']).value_counts()
            colors = {
                'Critical Risk': 'red',
                'High Risk': 'orange',
                'Medium Risk': 'yellow',
                'Lower Risk': 'green'
            }
            wedges, texts, autotexts = ax5.pie(risk_counts,
                                               labels=risk_counts.index,
                                               autopct='%1.1f%%',
                                               colors=[colors[x] for x in risk_counts.index],
                                               explode=[0.1 if x == 'Critical Risk' else 0
                                                        for x in risk_counts.index])
            ax5.set_title('Risk Level Distribution', fontsize=12, pad=15)
            # Make percentage labels more readable
            plt.setp(autotexts, size=8, weight="bold")
            plt.setp(texts, size=8)

            # 6. Temporal Distribution (if Year_Range is available)
            ax6 = fig.add_subplot(gs[2, 0:2])
            year_ranges = [range.split('-') for range in cluster_profiles['Year_Range']]
            year_spans = [(int(end) - int(start)) for start, end in year_ranges]
            years_start = [int(range[0]) for range in year_ranges]
            ax6.barh(range(len(cluster_profiles)), year_spans, left=years_start)
            ax6.set_title('Temporal Distribution of Clusters', fontsize=12, pad=15)
            ax6.set_xlabel('Year')
            ax6.set_ylabel('Cluster')
            ax6.set_yticks(range(len(cluster_profiles)))
            ax6.set_yticklabels([f'Cluster {i}' for i in range(len(cluster_profiles))])

            # 7. Performance Metrics Summary
            ax7 = fig.add_subplot(gs[2, 2])
            ax7.axis('off')
            metrics_text = (
                f"Clustering Performance Metrics:\n\n"
                f"Silhouette Score: {self.metrics['silhouette']:.3f}\n"
                f"Calinski-Harabasz: {self.metrics['calinski_harabasz']:.1f}\n"
                f"Stability Score: {self.metrics['stability_score']:.3f}\n"
                f"Stability Std: {self.metrics['stability_std']:.3f}\n\n"
                f"Total Accidents: {cluster_profiles['Size'].sum():,}\n"
                f"Overall Survival Rate: {cluster_profiles['Avg_Survival_Rate'].mean():.1%}"
            )
            ax7.text(0.05, 0.95, metrics_text,
                     transform=ax7.transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            traceback.print_exc()

    def _determine_key_characteristic(self, stats):
        """
        Determine the key characteristic of a cluster.

        Parameters:
            stats (pd.Series): Cluster statistics

        Returns:
            str: Key characteristic
        """
        try:
            characteristics = []

            # Add characteristics with priorities
            if stats['Military_Percentage'] > 75:
                characteristics.append(("Military Operations", 3))
            elif stats['Military_Percentage'] > 25:
                characteristics.append(("Mixed Operations", 1))

            if stats['Avg_Survival_Rate'] > 0.5:
                characteristics.append(("High Survival", 2))
            elif stats['Avg_Survival_Rate'] < 0.1:
                characteristics.append(("Critical Survival", 2))

            if stats['Avg_Aboard'] > 40:
                characteristics.append(("Large Aircraft", 1))
            elif stats['Avg_Aboard'] < 10:
                characteristics.append(("Small Aircraft", 1))

            if not characteristics:
                return "Standard Operations"

            # Return the characteristic with highest priority
            return max(characteristics, key=lambda x: x[1])[0]

        except Exception as e:
            print(f"Error determining key characteristic: {str(e)}")
            return "Unknown"


    def optimize_features(self, X, original_data):
        """
        Optimize feature selection for better clustering performance.

        Parameters:
            X (np.ndarray): Original scaled features
            original_data (pd.DataFrame): Original dataset

        Returns:
            np.ndarray: Optimized feature matrix
        """
        try:
            # Apply PCA to reduce noise and improve cluster separation
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            X_pca = pca.fit_transform(X)

            # Calculate feature importance
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i + 1}' for i in range(pca.components_.shape[0])],
                index=self.features
            )

            # Select most important features
            important_features = feature_importance.abs().sum(axis=1).sort_values(ascending=False)
            print("\nFeature Importance:")
            for feat, imp in important_features.items():
                print(f"{feat}: {imp:.3f}")

            return X_pca

        except Exception as e:
            print(f"Error in feature optimization: {str(e)}")
            return X

    def enhance_clustering(self, X, n_clusters=5):
        """
        Enhance clustering performance through parameter optimization.
        """
        try:
            best_score = -1
            best_params = {}
            best_metrics = None

            # Parameters to try
            init_methods = ['k-means++', 'random']
            n_inits = [10, 20, 30]
            max_iters = [300, 500]

            for init in init_methods:
                for n_init in n_inits:
                    for max_iter in max_iters:
                        kmeans = KMeans(
                            n_clusters=n_clusters,
                            init=init,
                            n_init=n_init,
                            max_iter=max_iter,
                            random_state=42
                        )
                        labels = kmeans.fit_predict(X)
                        score = silhouette_score(X, labels)

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'init': init,
                                'n_init': n_init,
                                'max_iter': max_iter
                            }
                            self.kmeans = kmeans
                            # Store metrics
                            self.metrics = {
                                'silhouette': score,
                                'calinski_harabasz': calinski_harabasz_score(X, labels),
                                'davies_bouldin': davies_bouldin_score(X, labels),
                                'stability_score': score,  # Using silhouette as initial stability
                                'stability_std': 0.05  # Default value
                            }

                            # Calculate stability through cross-validation
                            kf = KFold(n_splits=5, shuffle=True, random_state=42)
                            stability_scores = []
                            for train_idx, val_idx in kf.split(X):
                                X_val = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
                                val_labels = kmeans.predict(X_val)
                                if len(np.unique(val_labels)) > 1:  # Check if more than one cluster
                                    stability_scores.append(silhouette_score(X_val, val_labels))

                            if stability_scores:
                                self.metrics['stability_score'] = np.mean(stability_scores)
                                self.metrics['stability_std'] = np.std(stability_scores)

            print("\nBest Parameters Found:")
            for param, value in best_params.items():
                print(f"{param}: {value}")
            print(f"Best Silhouette Score: {best_score:.3f}")

            return self.metrics

        except Exception as e:
            print(f"Error in clustering enhancement: {str(e)}")
            return None

    def create_visualizations(self, cluster_profiles, report):
        """
        Create comprehensive visualizations with temporal trends and risk patterns.
        """
        try:
            # Create figure with GridSpec
            plt.style.use('default')
            fig = plt.figure(figsize=(20, 20))  # Increased figure size
            gs = GridSpec(4, 3, figure=fig)  # Added an extra row

            # 1. Cluster Size Distribution
            ax1 = fig.add_subplot(gs[0, 0])
            cluster_sizes = cluster_profiles['Size']
            bars = ax1.bar(range(len(cluster_sizes)), cluster_sizes)
            ax1.set_title('Cluster Size Distribution', fontsize=12, pad=15)
            ax1.set_xlabel('Cluster')
            ax1.set_ylabel('Number of Accidents')
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height):,}',
                         ha='center', va='bottom')

            # 2. Survival Rates Analysis
            ax2 = fig.add_subplot(gs[0, 1])
            survival_data = pd.DataFrame({
                'Survival Rate': cluster_profiles['Avg_Survival_Rate'],
                'Fatal Rate': 1 - cluster_profiles['Avg_Survival_Rate']
            }).T
            sns.heatmap(survival_data,
                        annot=True,
                        fmt='.1%',
                        cmap='RdYlGn',
                        ax=ax2)
            ax2.set_title('Survival vs Fatality Rates')

            # 3. Military vs Civilian Distribution
            ax3 = fig.add_subplot(gs[0, 2])
            military_data = pd.DataFrame({
                'Military': cluster_profiles['Military_Percentage'],
                'Civilian': 100 - cluster_profiles['Military_Percentage']
            })
            military_data.plot(kind='bar',
                               stacked=True,
                               ax=ax3,
                               color=['navy', 'lightblue'])
            ax3.set_title('Military vs Civilian Distribution')
            ax3.set_xlabel('Cluster')
            ax3.set_ylabel('Percentage')
            ax3.legend(loc='upper right')

            # 4. Risk Assessment Matrix with Enhanced Annotations
            ax4 = fig.add_subplot(gs[1, :2])
            scatter = ax4.scatter(cluster_profiles['Avg_Aboard'],
                                  cluster_profiles['Avg_Survival_Rate'],
                                  s=cluster_profiles['Size'] / 10,
                                  c=cluster_profiles['Military_Percentage'],
                                  cmap='coolwarm')
            for i, row in cluster_profiles.iterrows():
                ax4.annotate(
                    f'Cluster {i}\n'
                    f'Size: {row["Size"]:,}\n'
                    f'Survival: {row["Avg_Survival_Rate"]:.1%}\n'
                    f'Military: {row["Military_Percentage"]:.1f}%',
                    (row['Avg_Aboard'], row['Avg_Survival_Rate']),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                    fontsize=8
                )
            ax4.set_title('Risk Assessment Matrix')
            ax4.set_xlabel('Average Passengers Aboard')
            ax4.set_ylabel('Survival Rate')
            plt.colorbar(scatter, ax=ax4, label='Military Percentage')

            # 5. Risk Level Distribution with Enhanced Legend
            ax5 = fig.add_subplot(gs[1, 2])
            risk_counts = pd.Series(report['risk_levels']).value_counts()
            colors = {
                'Critical Risk': 'red',
                'High Risk': 'orange',
                'Medium Risk': 'yellow',
                'Lower Risk': 'green'
            }
            wedges, texts, autotexts = ax5.pie(risk_counts,
                                               labels=risk_counts.index,
                                               autopct='%1.1f%%',
                                               colors=[colors[x] for x in risk_counts.index],
                                               explode=[0.1 if x == 'Critical Risk' else 0.05
                                                        for x in risk_counts.index])
            ax5.set_title('Risk Level Distribution')
            plt.setp(autotexts, size=8, weight="bold")
            plt.setp(texts, size=8)

            # 6. Temporal Distribution
            ax6 = fig.add_subplot(gs[2, :])
            year_ranges = [range.split('-') for range in cluster_profiles['Year_Range']]
            for i, (start, end) in enumerate(year_ranges):
                start_year, end_year = int(start), int(end)
                ax6.barh(i, end_year - start_year, left=start_year,
                         height=0.3, label=f'Cluster {i}')
                ax6.text(start_year, i, f'{start_year}', ha='right', va='center')
                ax6.text(end_year, i, f'{end_year}', ha='left', va='center')
            ax6.set_title('Temporal Coverage by Cluster')
            ax6.set_xlabel('Year')
            ax6.set_yticks(range(len(cluster_profiles)))
            ax6.set_yticklabels([f'Cluster {i}' for i in range(len(cluster_profiles))])
            ax6.grid(True, alpha=0.3)

            # 7. Performance Metrics and Statistical Summary
            ax7 = fig.add_subplot(gs[3, :])
            ax7.axis('off')
            metrics_text = (
                f"Clustering Performance Metrics:\n\n"
                f"Silhouette Score: {self.metrics['silhouette']:.3f} "
                f"(Quality: {'Poor' if self.metrics['silhouette'] < 0.2 else 'Fair' if self.metrics['silhouette'] < 0.4 else 'Good'})\n"
                f"Calinski-Harabasz: {self.metrics['calinski_harabasz']:.1f}\n"
                f"Stability Score: {self.metrics['stability_score']:.3f} "
                f"(±{self.metrics['stability_std']:.3f})\n\n"
                f"Key Findings:\n"
                f"• Total Accidents: {cluster_profiles['Size'].sum():,}\n"
                f"• Overall Survival Rate: {cluster_profiles['Avg_Survival_Rate'].mean():.1%}\n"
                f"• Military Involvement: {cluster_profiles['Military_Percentage'].mean():.1f}%\n"
                f"• Most Critical Cluster: Cluster {cluster_profiles['Avg_Survival_Rate'].idxmin()} "
                f"({cluster_profiles['Avg_Survival_Rate'].min():.1%} survival rate)\n"
                f"• Best Performing Cluster: Cluster {cluster_profiles['Avg_Survival_Rate'].idxmax()} "
                f"({cluster_profiles['Avg_Survival_Rate'].max():.1%} survival rate)"
            )
            ax7.text(0.05, 0.95, metrics_text,
                     transform=ax7.transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            traceback.print_exc()


    def analyze_temporal_trends(self, cluster_profiles):
        """
        Analyze temporal evolution of clusters.
        """
        try:
            temporal_analysis = {
                'year_range': {},
                'trends': {},
                'evolution': {}
            }

            for idx, row in cluster_profiles.iterrows():
                start_year, end_year = map(int, row['Year_Range'].split('-'))
                span = end_year - start_year

                temporal_analysis['year_range'][f'Cluster_{idx}'] = {
                    'span': span,
                    'start': start_year,
                    'end': end_year,
                    'concentration': span / len(range(1908, 2009)),  # Normalized time span
                    'era_classification': self._classify_aviation_era(start_year, end_year)
                }

                # Analyze survival rate trends
                temporal_analysis['trends'][f'Cluster_{idx}'] = {
                    'modernization_impact': self._calculate_modernization_impact(start_year, end_year,
                                                                                 row['Avg_Survival_Rate']),
                    'temporal_density': row['Size'] / span  # Accidents per year
                }

            return temporal_analysis

        except Exception as e:
            print(f"Error in temporal analysis: {str(e)}")
            return None


    def _classify_aviation_era(self, start_year, end_year):
        """
        Classify the aviation era for temporal analysis.
        """
        eras = {
            (1908, 1945): "Early Aviation",
            (1946, 1969): "Propeller Era",
            (1970, 1989): "Early Jet Age",
            (1990, 2009): "Modern Aviation"
        }

        primary_era = None
        coverage = {}

        for (era_start, era_end), era_name in eras.items():
            overlap_start = max(start_year, era_start)
            overlap_end = min(end_year, era_end)

            if overlap_end >= overlap_start:
                overlap = overlap_end - overlap_start + 1
                total_span = end_year - start_year + 1
                coverage[era_name] = overlap / total_span

                if primary_era is None or coverage[era_name] > coverage[primary_era]:
                    primary_era = era_name

        return {
            'primary_era': primary_era,
            'era_coverage': coverage
        }


    def _calculate_modernization_impact(self, start_year, end_year, survival_rate):
        """
        Calculate the impact of aviation modernization on survival rates.
        """
        modern_aviation_start = 1990

        if end_year < modern_aviation_start:
            return "Pre-modern"

        if start_year >= modern_aviation_start:
            return "Modern"

        # Calculate the proportion of modern era
        total_span = end_year - start_year
        modern_span = end_year - max(start_year, modern_aviation_start)
        modern_proportion = modern_span / total_span

        impact_score = survival_rate * (1 + modern_proportion)

        if impact_score > 0.7:
            return "Strong positive"
        elif impact_score > 0.4:
            return "Moderate positive"
        else:
            return "Limited impact"


    def calculate_risk_score(self, cluster_stats):
        """
        Calculate comprehensive risk score with enhanced metrics.
        """
        try:
            # Base risk factors
            survival_factor = 1 - cluster_stats['Avg_Survival_Rate']
            fatality_factor = cluster_stats['Fatal_Accidents'] / cluster_stats['Size']
            zero_survival_factor = cluster_stats['Zero_Survival_Rate'] / cluster_stats['Size']

            # Additional risk factors
            passenger_load_factor = min(cluster_stats['Avg_Aboard'] / 100, 1)
            military_factor = cluster_stats['Military_Percentage'] / 100

            # Weighted risk score calculation
            risk_score = (
                    survival_factor * 0.35 +  # Survival weight
                    fatality_factor * 0.25 +  # Fatality weight
                    zero_survival_factor * 0.15 +  # Zero survival weight
                    passenger_load_factor * 0.15 +  # Passenger capacity weight
                    military_factor * 0.10  # Military operations weight
            )

            # Scale to 0-100 and add risk categories
            scaled_score = risk_score * 100
            risk_category = (
                "Extreme Risk" if scaled_score > 80 else
                "High Risk" if scaled_score > 60 else
                "Moderate Risk" if scaled_score > 40 else
                "Low Risk"
            )

            return {
                'score': scaled_score,
                'category': risk_category,
                'components': {
                    'survival_impact': survival_factor * 0.35 * 100,
                    'fatality_impact': fatality_factor * 0.25 * 100,
                    'zero_survival_impact': zero_survival_factor * 0.15 * 100,
                    'passenger_impact': passenger_load_factor * 0.15 * 100,
                    'military_impact': military_factor * 0.10 * 100
                }
            }

        except Exception as e:
            print(f"Error calculating risk score: {str(e)}")
            return None


    def compare_clusters(self, cluster_profiles):
        """
        Perform enhanced comparative analysis between clusters.
        """
        try:
            comparisons = {}
            best_performer = cluster_profiles.loc[cluster_profiles['Avg_Survival_Rate'].idxmax()]

            for idx, row in cluster_profiles.iterrows():
                if idx != best_performer.name:
                    # Calculate gaps and potential improvements
                    survival_gap = best_performer['Avg_Survival_Rate'] - row['Avg_Survival_Rate']
                    potential_lives = int(row['Fatal_Accidents'] * survival_gap)
                    relative_risk = row['Fatal_Accidents'] / row['Size'] / (
                                best_performer['Fatal_Accidents'] / best_performer['Size'])

                    improvement_metrics = {
                        'survival_gap': survival_gap,
                        'potential_lives_saved': potential_lives,
                        'relative_risk_ratio': relative_risk,
                        'improvement_priority': 'Critical' if survival_gap > 0.6 else 'High' if survival_gap > 0.4 else 'Medium' if survival_gap > 0.2 else 'Low',
                        'recommended_actions': self._generate_improvement_actions(survival_gap, row)
                    }

                    comparisons[f'Cluster_{idx}'] = improvement_metrics

            return comparisons

        except Exception as e:
            print(f"Error in cluster comparison: {str(e)}")
            return None


    def _generate_improvement_actions(self, survival_gap, cluster_stats):
        """
        Generate specific improvement actions based on cluster characteristics.
        """
        actions = []

        if survival_gap > 0.6:
            actions.extend([
                "Immediate comprehensive safety review",
                "Emergency response protocol overhaul",
                "Mandatory advanced safety training"
            ])
        elif survival_gap > 0.4:
            actions.extend([
                "Enhanced safety protocols implementation",
                "Regular safety audits",
                "Crew training program upgrade"
            ])

        if cluster_stats['Military_Percentage'] > 50:
            actions.extend([
                "Military-specific safety measures",
                "Combat mission risk assessment",
                "Specialized emergency procedures"
            ])

        if cluster_stats['Avg_Aboard'] > 40:
            actions.extend([
                "Large aircraft evacuation protocols",
                "Passenger safety system upgrade",
                "Enhanced emergency equipment"
            ])

        return actions


def main():
    """Main execution function with enhanced analysis capabilities."""
    clustering = AviationRiskClustering()

    try:
        print("Initiating Enhanced Aviation Risk Analysis...")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load and preprocess data
        data = clustering.load_and_preprocess_data('Airplane_Crashes_and_Fatalities_Since_1908.csv')
        if data is None:
            raise ValueError("Failed to load and preprocess data")
        print(f"Processed {len(data):,} accident records")

        # Scale features
        X = clustering.scaler.fit_transform(data)

        # Optimize features
        print("\nOptimizing features...")
        X_optimized = clustering.optimize_features(X, data)

        # Find optimal number of clusters
        optimal_k_results = clustering.find_optimal_k(X_optimized)
        print("\nClustering Performance by K:")
        print(optimal_k_results)

        # Enhanced clustering
        print("\nEnhancing clustering performance...")
        clustering.enhance_clustering(X_optimized, n_clusters=5)

        # Analyze clusters
        cluster_profiles = clustering.analyze_clusters(X_optimized, data)
        if cluster_profiles is None:
            raise ValueError("Cluster analysis failed")

        # Temporal analysis
        print("\nAnalyzing temporal trends...")
        temporal_analysis = clustering.analyze_temporal_trends(cluster_profiles)

        # Calculate risk scores
        print("\nCalculating risk scores...")
        risk_scores = {f"Cluster_{i}": clustering.calculate_risk_score(row)
                       for i, row in cluster_profiles.iterrows()}

        # Comparative analysis
        print("\nPerforming comparative analysis...")
        comparisons = clustering.compare_clusters(cluster_profiles)

        # Generate and print report
        report = clustering.generate_report(cluster_profiles)
        if report is None:
            raise ValueError("Report generation failed")

        # Update report with new analyses
        report.update({
            'temporal_analysis': temporal_analysis,
            'risk_scores': risk_scores,
            'cluster_comparisons': comparisons
        })

        # Print enhanced report
        clustering.print_report(report)

        # Create visualizations
        print("\nCreating enhanced visualizations...")
        clustering.create_visualizations(cluster_profiles, report)

        # Print execution summary
        end_time = datetime.now()
        execution_time = (end_time - datetime.strptime(
            start_time := datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '%Y-%m-%d %H:%M:%S'
        )).total_seconds()

        print(f"\nEnhanced analysis completed in {execution_time:.2f} seconds")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()
