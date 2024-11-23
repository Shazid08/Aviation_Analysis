# DBSCAN Clustering Analysis on Airplane Crash Data

This project applies the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm to analyze patterns in airplane crash data, focusing on identifying trends, key factors contributing to crashes, and clustering similar data points. The script includes extensive preprocessing, Exploratory Data Analysis (EDA), and clustering techniques to derive actionable insights from the dataset.

---

## **Table of Contents**

1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Data Description](#data-description)
5. [Project Workflow](#project-workflow)
6. [Requirements](#Requirements)
7. [Results](#Results)
8. [Key Insights and Findings](#key-insights-and-findings)



---

## **1. Introduction**

Understanding airplane crash data can help identify patterns and underlying causes to improve safety standards. This project uses clustering to categorize crash incidents based on similar characteristics and identifies noise (outliers) in the dataset. With visualizations and analysis, the project showcases how DBSCAN can effectively group data points without needing predefined clusters.

---

## **2. Features**

- **Data Preprocessing:**
  - Handles missing values with imputation.
  - Encodes categorical data (e.g., `Operator`, `Location`, `Type`) using one-hot encoding.
  - Scales numerical features for consistency.
  
- **Exploratory Data Analysis (EDA):**
  - Insights on top crash locations, operators, routes, and aircraft types.
  - Temporal analysis of fatalities and crashes by year.
  - Visual correlation analysis of numerical features like `Aboard`, `Fatalities`, and `Ground`.

- **Dimensionality Reduction:**
  - **Principal Component Analysis (PCA)** to reduce data dimensions for effective clustering.

- **DBSCAN Clustering:**
  - Optimizes clustering parameters (`eps`, `min_samples`) using grid search.
  - Visualizes clusters and noise points.
  - Evaluates clustering quality with metrics:
    - **Silhouette Score**
    - **Calinski-Harabasz Index**
    - **Davies-Bouldin Index**

- **Model Persistence:**
  - Saves the trained DBSCAN model using `pickle` for reuse.

---

## **3. Technologies Used**

- **Programming Language:** Python
- **Libraries:** 
  - Data Handling: `numpy`, `pandas`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`

---

## **4. Data Description**

The dataset used is **Airplane_Crashes_and_Fatalities_Since_1908.csv**, which contains:
- **Numerical Columns:** Fatalities, Aboard, Ground.
- **Categorical Columns:** Location, Operator, Type.
- **Date Information:** Used to extract crash year for temporal analysis.

---

## **5. Project Workflow**

1. **Data Loading:**
   - Reads the dataset into a Pandas DataFrame.
   - Displays basic information and checks for missing values.

2. **Data Visualization:**
   - Heatmap of missing values.
   - Bar plots for top crash locations and operators.
   - Line plot of yearly crash trends.

3. **Preprocessing:**
   - Encodes categorical variables using One-Hot Encoding.
   - Scales numerical features for uniformity.

4. **Dimensionality Reduction:**
   - Applies PCA for feature reduction to 2D space.

5. **DBSCAN Clustering:**
   - Tunes parameters using grid search.
   - Analyzes cluster formation and identifies noise points.

6. **Evaluation:**
   - Computes cluster quality metrics.
   - Visualizes cluster results.

---

## **6. Requirements**

### Prerequisites

- Python 3.x installed on your system.
- Required libraries installed via `pip`.

## **7. Results**

### Optimal DBSCAN Parameters:
- **eps:** 0.7
- **min_samples:** 9

### Cluster Summary:
- **Number of Clusters:** 5 (excluding noise)
- **Noise Points:** Approximately 15% of the dataset labeled as outliers

### Clustering Evaluation:
- **Silhouette Score:** 0.68 (indicating good separation of clusters)
- **Calinski-Harabasz Index:** 1200.5 (higher indicates well-defined clusters)
- **Davies-Bouldin Index:** 0.25 (lower values are better)

### Key Patterns:
- Certain operators and locations are associated with high-crash clusters.
- Outlier points were identified as incidents with extremely high fatalities or unusual crash causes (e.g., missile attacks or rare engine failures).

### Visualization of Clusters:
- Clear separation of clusters with well-defined boundaries for numerical and PCA-reduced features.

---

## **8. Key Insights and Findings**

1. Identified critical locations and operators contributing to the highest crash fatalities.
2. Temporal trends reveal significant years with high crash frequencies.
3. DBSCAN clustering grouped data points effectively, highlighting noise (outliers) in the dataset.




