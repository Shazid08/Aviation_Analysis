import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats


def create_sample_data(n_samples=1000):
    np.random.seed(42)

    dates = pd.date_range(start='1980-01-01', end='2000-12-31', periods=n_samples)

    # Operator types with realistic distributions
    operator_types = [
        'Commercial Airline', 'Military Transport', 'Private Charter',
        'Cargo Airline', 'Military Fighter', 'Regional Airline'
    ]

    data = {
        'Date': dates,
        'Time': [f"{np.random.randint(0, 24):02d}:00" for _ in range(n_samples)],
        'Operator': np.random.choice(operator_types, size=n_samples,
                                     p=[0.4, 0.15, 0.15, 0.15, 0.05, 0.1]),
        'Aboard': [],
        'Fatalities': [],
        'Ground': []
    }

    # Generate correlated data
    for _ in range(n_samples):
        operator = data['Operator'][_]

        # Aircraft size based on operator
        if operator == 'Commercial Airline':
            aboard = int(np.random.normal(180, 30, 1)[0])
        elif operator == 'Military Transport':
            aboard = int(np.random.normal(100, 20, 1)[0])
        elif operator == 'Private Charter':
            aboard = int(np.random.normal(15, 5, 1)[0])
        elif operator == 'Cargo Airline':
            aboard = int(np.random.normal(5, 2, 1)[0])
        elif operator == 'Military Fighter':
            aboard = int(np.random.normal(2, 1, 1)[0])
        else:  # Regional Airline
            aboard = int(np.random.normal(70, 15, 1)[0])

        aboard = max(1, aboard)
        data['Aboard'].append(aboard)

        # Generate fatalities
        survival_rate = np.random.beta(2, 2)
        fatalities = int(aboard * (1 - survival_rate))
        data['Fatalities'].append(min(fatalities, aboard))

        # Generate ground casualties
        data['Ground'].append(np.random.poisson(1))

    df = pd.DataFrame(data)

    # Add derived features
    df['IsMilitary'] = df['Operator'].str.contains('Military').astype(int)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Hour'] = df['Time'].str.split(':').str[0].astype(int)
    df['Survival_Rate'] = (df['Aboard'] - df['Fatalities']) / df['Aboard']
    df['Total_Casualties'] = df['Fatalities'] + df['Ground']
    df['Casualty_Rate'] = df['Total_Casualties'] / df['Aboard']

    return df


def analyze_clusters(kmeans, data):
    """
    Analyze clusters and generate profiles
    """
    labels = kmeans.labels_
    profiles = []

    for i in range(kmeans.n_clusters):
        mask = labels == i
        cluster_data = data[mask]

        profile = {
            'Cluster': i,
            'Size': len(cluster_data),
            'Avg_Survival_Rate': cluster_data['Survival_Rate'].mean(),
            'Avg_Casualties': cluster_data['Total_Casualties'].mean(),
            'Military_Percentage': cluster_data['IsMilitary'].mean() * 100,
            'Avg_Aboard': cluster_data['Aboard'].mean()
        }
        profiles.append(profile)

    return pd.DataFrame(profiles)


def plot_results(data, kmeans, cluster_profiles):
    """
    Create basic visualizations
    """
    plt.figure(figsize=(15, 5))

    # Plot 1: Cluster Sizes
    plt.subplot(131)
    plt.bar(range(len(cluster_profiles)), cluster_profiles['Size'])
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cases')

    # Plot 2: Survival Rates
    plt.subplot(132)
    plt.bar(range(len(cluster_profiles)), cluster_profiles['Avg_Survival_Rate'])
    plt.title('Average Survival Rates by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Survival Rate')

    # Plot 3: Scatter Plot
    plt.subplot(133)
    plt.scatter(data['Aboard'], data['Survival_Rate'], c=kmeans.labels_, cmap='viridis')
    plt.title('Clusters: Aboard vs Survival Rate')
    plt.xlabel('Passengers Aboard')
    plt.ylabel('Survival Rate')

    plt.tight_layout()
    plt.show()


def run_demo():
    print("=== Aviation Risk Analysis Demo ===")

    # Create sample data
    print("\n1. Creating sample dataset...")
    df = create_sample_data()
    print(f"Created dataset with {len(df)} records")
    print("\nSample of the data:")
    print(df.head())

    # Prepare features for clustering
    print("\n2. Preparing features...")
    features = ['Survival_Rate', 'Casualty_Rate', 'IsMilitary', 'Aboard']
    X = StandardScaler().fit_transform(df[features])

    # Perform clustering
    print("\n3. Performing clustering analysis...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)

    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)

    print("\nClustering Metrics:")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Score: {calinski:.3f}")
    print(f"Davies-Bouldin Score: {davies:.3f}")

    # Analyze clusters
    print("\n4. Analyzing clusters...")
    cluster_profiles = analyze_clusters(kmeans, df)
    print("\nCluster Profiles:")
    print(cluster_profiles)

    # Create visualizations
    print("\n5. Creating visualizations...")
    plot_results(df, kmeans, cluster_profiles)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()