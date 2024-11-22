import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class AviationSurvivalAnalyzer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.features = ['year', 'aboard', 'ground', 'fatalities_ratio', 'ground_to_aboard_ratio']
        self.target = 'survival_rate'

    def analyze_fatalities_relationship(self):
        """Analyze the relationship between fatalities and survival rate"""
        plt.figure(figsize=(15, 10))

        # Create subplot grid
        gs = plt.GridSpec(2, 2)

        # 1. Scatter plot with regression line
        ax1 = plt.subplot(gs[0, :])
        sns.regplot(data=self.data,
                    x='fatalities_ratio',
                    y=self.target,
                    scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red'})
        plt.title('Survival Rate vs Fatalities Ratio')
        plt.xlabel('Fatalities Ratio')
        plt.ylabel('Survival Rate')

        # Calculate correlation statistics
        correlation = stats.pearsonr(self.data['fatalities_ratio'],
                                     self.data[self.target])
        plt.text(0.05, 0.95,
                 f'Correlation: {correlation[0]:.4f}\np-value: {correlation[1]:.4e}',
                 transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        # 2. Fatalities ratio distribution
        ax2 = plt.subplot(gs[1, 0])
        sns.histplot(data=self.data, x='fatalities_ratio', bins=30)
        plt.title('Distribution of Fatalities Ratio')
        plt.xlabel('Fatalities Ratio')

        # 3. Custom binning for fatalities ratio
        ax3 = plt.subplot(gs[1, 1])
        # Create manual bins to avoid duplicate edges
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        self.data['fatality_category'] = pd.cut(self.data['fatalities_ratio'],
                                                bins=bins,
                                                labels=labels)

        avg_survival = self.data.groupby('fatality_category')[self.target].mean()
        avg_survival.plot(kind='bar')
        plt.title('Average Survival Rate by Fatalities Category')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def temporal_trend_analysis(self):
        """Analyze temporal trends in survival rates"""
        plt.figure(figsize=(15, 12))

        # 1. Time series plot with moving average
        ax1 = plt.subplot(311)
        yearly_avg = self.data.groupby('year')[self.target].mean()
        yearly_avg.plot(style='o-', alpha=0.5, label='Yearly Average')

        # Add 5-year moving average
        ma = yearly_avg.rolling(window=5, min_periods=1).mean()
        ma.plot(style='r-', linewidth=2, label='5-Year Moving Average')

        plt.title('Survival Rate Trends Over Time')
        plt.xlabel('Year')
        plt.ylabel('Average Survival Rate')
        plt.legend()

        # 2. Decade analysis
        ax2 = plt.subplot(312)
        self.data['decade'] = (self.data['year'] // 10) * 10
        decade_stats = self.data.groupby('decade').agg({
            self.target: ['mean', 'std', 'count']
        }).round(4)

        decade_stats[self.target]['mean'].plot(kind='bar',
                                               yerr=decade_stats[self.target]['std'],
                                               capsize=5)
        plt.title('Average Survival Rate by Decade (with Standard Deviation)')
        plt.xlabel('Decade')
        plt.ylabel('Average Survival Rate')

        # 3. Recent trends
        ax3 = plt.subplot(313)
        recent_data = self.data[self.data['year'] >= self.data['year'].max() - 10]
        sns.regplot(data=recent_data, x='year', y=self.target,
                    scatter_kws={'alpha': 0.5})
        plt.title('Survival Rate Trend in Last 10 Years')
        plt.xlabel('Year')
        plt.ylabel('Survival Rate')

        plt.tight_layout()
        plt.show()

    def analyze_size_impact(self):
        """Analyze the relationship between aircraft size and survival"""
        plt.figure(figsize=(15, 10))

        # 1. Scatter plot
        ax1 = plt.subplot(221)
        plt.scatter(self.data['aboard'], self.data[self.target], alpha=0.5)
        plt.title('Survival Rate vs Number of People Aboard')
        plt.xlabel('Number Aboard')
        plt.ylabel('Survival Rate')

        # 2. Size categories using custom bins
        size_bins = [0, 10, 50, 100, 200, float('inf')]
        size_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
        self.data['size_category'] = pd.cut(self.data['aboard'],
                                            bins=size_bins,
                                            labels=size_labels)

        ax2 = plt.subplot(222)
        sns.boxplot(data=self.data, x='size_category', y=self.target)
        plt.title('Survival Rate Distribution by Aircraft Size')
        plt.xticks(rotation=45)

        # 3. Size and fatalities relationship
        ax3 = plt.subplot(212)
        size_stats = self.data.groupby('size_category').agg({
            self.target: 'mean',
            'fatalities_ratio': 'mean',
            'aboard': 'count'
        }).round(4)

        # Dual axis plot
        ax3_twin = ax3.twinx()

        bars1 = ax3.bar(np.arange(len(size_stats)) - 0.2,
                        size_stats[self.target],
                        width=0.4,
                        color='blue',
                        label='Survival Rate')
        bars2 = ax3_twin.bar(np.arange(len(size_stats)) + 0.2,
                             size_stats['fatalities_ratio'],
                             width=0.4,
                             color='red',
                             label='Fatalities Ratio')

        ax3.set_xticks(range(len(size_stats)))
        ax3.set_xticklabels(size_stats.index, rotation=45)
        ax3.set_ylabel('Average Survival Rate', color='blue')
        ax3_twin.set_ylabel('Average Fatalities Ratio', color='red')
        plt.title('Survival Rate and Fatalities Ratio by Aircraft Size')

        # Add legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.show()

    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "=" * 60)
        print(" Advanced Survival Rate Analysis ".center(60, "="))
        print("=" * 60 + "\n")

        # Basic statistics
        print("Basic Statistics:")
        print(f"  Total Incidents:      {len(self.data)}")
        print(f"  Mean Survival Rate:   {self.data[self.target].mean():.4f}")
        print(f"  Median Survival Rate: {self.data[self.target].median():.4f}")
        print(f"  Std Deviation:        {self.data[self.target].std():.4f}")

        # Feature correlations
        print("\nFeature Correlations with Survival Rate:")
        correlations = self.data[self.features + [self.target]].corr()[self.target]
        for feature, corr in correlations.sort_values(ascending=False).items():
            if feature != self.target:
                print(f"  {feature:20}: {corr:8.4f}")

        # Recent trends
        recent_data = self.data[self.data['year'] >= self.data['year'].max() - 10]
        print("\nRecent Trends (Last 10 Years):")
        print(f"  Average Survival Rate: {recent_data[self.target].mean():.4f}")
        print(
            f"  Trend Direction:       {'Increasing' if recent_data.groupby('year')[self.target].mean().diff().mean() > 0 else 'Decreasing'}")


# Run analysis
if __name__ == "__main__":
    try:
        analyzer = AviationSurvivalAnalyzer('Enhanced_Airplane_Crashes.csv')
        analyzer.analyze_fatalities_relationship()
        analyzer.temporal_trend_analysis()
        analyzer.analyze_size_impact()
        analyzer.print_analysis_summary()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()