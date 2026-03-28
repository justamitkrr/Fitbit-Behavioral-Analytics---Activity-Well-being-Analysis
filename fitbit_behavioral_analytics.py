#!/usr/bin/env python3
"""
Fitbit Behavioral Analytics - Complete Implementation
Analyzes user activity, sleep, and mood patterns to identify behavioral correlations

Author: Data Analyst
Date: 2024
Purpose: Uncover activity-mood relationships for personalized engagement
"""

# ============================================================================
# PART 1: IMPORTS AND CONFIGURATION
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2contingency
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("=" * 70)
print("FITBIT BEHAVIORAL ANALYTICS - INITIALIZATION")
print("=" * 70)
print("\n✓ Libraries imported successfully")

# ============================================================================
# PART 2: DATA GENERATION (Use real data if available)
# ============================================================================

def create_sample_fitbit_data(n_users=50, days_per_user=30):
    """
    Generate realistic synthetic Fitbit data
    
    Parameters:
    - n_users: Number of unique users
    - days_per_user: Number of days of data per user
    """
    np.random.seed(42)
    
    print("\n[GENERATING SAMPLE DATA]")
    
    data = []
    start_date = pd.Timestamp('2024-01-01')
    
    for user_id in range(1, n_users + 1):
        # User-specific baseline behavior
        user_activity_level = np.random.choice(['low', 'medium', 'high'], 
                                               p=[0.3, 0.5, 0.2])
        
        if user_activity_level == 'low':
            base_steps = np.random.randint(2000, 5000)
        elif user_activity_level == 'medium':
            base_steps = np.random.randint(5000, 10000)
        else:
            base_steps = np.random.randint(10000, 15000)
        
        for day in range(days_per_user):
            current_date = start_date + timedelta(days=day)
            
            # Generate steps with some randomness and weekly patterns
            daily_variation = np.random.normal(0, 2000)
            weekend_boost = 1000 if current_date.dayofweek >= 5 else 0
            steps = max(0, int(base_steps + daily_variation + weekend_boost))
            
            # Sleep hours (6-9 hours, normally distributed)
            sleep_hours = np.clip(np.random.normal(7, 1), 4, 10)
            
            # Calories (correlated with steps)
            base_calories = 1800
            activity_calories = steps * 0.04  # ~0.04 cal per step
            calories = int(base_calories + activity_calories + np.random.normal(0, 100))
            
            # Mood (1-5 scale, positively correlated with activity)
            # Higher steps -> higher mood tendency
            if steps < 3000:
                mood_prob = [0.3, 0.3, 0.25, 0.1, 0.05]  # Skewed negative
            elif steps < 6000:
                mood_prob = [0.15, 0.25, 0.35, 0.2, 0.05]
            elif steps < 10000:
                mood_prob = [0.05, 0.15, 0.30, 0.35, 0.15]
            else:
                mood_prob = [0.02, 0.08, 0.20, 0.45, 0.25]  # Skewed positive
            
            mood = np.random.choice([1, 2, 3, 4, 5], p=mood_prob)
            
            # Activity code (for demonstration of mapping)
            if steps < 5000:
                activity_code = '0'
            elif steps < 10000:
                activity_code = '500'
            elif steps < 15000:
                activity_code = '1000'
            else:
                activity_code = '1500'
            
            data.append({
                'User_ID': f'U{user_id:03d}',
                'Date': current_date,
                'Steps': steps,
                'Sleep_Hours': round(sleep_hours, 1),
                'Calories': calories,
                'Mood': mood,
                'Activity_Code': activity_code
            })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df):,} records for {n_users} users over {days_per_user} days")
    return df

def load_data(filepath=None):
    """Load Fitbit data from file or generate sample data"""
    print("\n[LOADING DATA]")
    
    if filepath:
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Data loaded from {filepath}: {df.shape[0]:,} rows")
            return df
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            print("  Creating sample dataset instead...")
    
    return create_sample_fitbit_data()

# ============================================================================
# PART 3: DATA CLEANING AND PREPROCESSING
# ============================================================================

def clean_data(df):
    """
    Clean and preprocess Fitbit data
    """
    print("\n[CLEANING DATA]")
    
    initial_count = len(df)
    
    # Convert date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Handle missing values
    critical_cols = ['Steps', 'Mood']
    df = df.dropna(subset=critical_cols)
    
    # Fill missing sleep data with median
    if 'Sleep_Hours' in df.columns:
        df['Sleep_Hours'].fillna(df['Sleep_Hours'].median(), inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove outliers using IQR
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]
    
    df = remove_outliers_iqr(df, 'Steps')
    if 'Calories' in df.columns:
        df = remove_outliers_iqr(df, 'Calories')
    
    final_count = len(df)
    removed = initial_count - final_count
    
    print(f"✓ Data cleaned: {removed:,} rows removed ({removed/initial_count*100:.1f}%)")
    print(f"✓ Final dataset: {final_count:,} rows")
    
    return df

# ============================================================================
# PART 4: FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Create new features using NumPy optimization
    """
    print("\n[ENGINEERING FEATURES]")
    
    # Convert to NumPy arrays for optimized computation
    steps_array = df['Steps'].values
    calories_array = df['Calories'].values if 'Calories' in df.columns else None
    sleep_array = df['Sleep_Hours'].values if 'Sleep_Hours' in df.columns else None
    
    # Compute aggregate metrics
    print("\nAggregate Metrics (NumPy):")
    print(f"  Steps - Mean: {np.mean(steps_array):.0f}, Std: {np.std(steps_array):.0f}")
    print(f"  Steps - Max: {np.max(steps_array)}, Min: {np.min(steps_array)}")
    print(f"  Steps - Median: {np.median(steps_array):.0f}")
    
    if sleep_array is not None:
        print(f"  Sleep - Mean: {np.mean(sleep_array):.2f} hours")
    
    if calories_array is not None:
        print(f"  Calories - Mean: {np.mean(calories_array):.0f}")
    
    # Map activity status
    def map_activity_status(steps):
        """Map step counts to meaningful activity labels"""
        if steps < 5000:
            return 'Inactive'
        elif steps < 10000:
            return 'Moderately Active'
        elif steps < 15000:
            return 'Active'
        else:
            return 'Highly Active'
    
    df['Activity_Status'] = df['Steps'].apply(map_activity_status)
    print("\n✓ Activity status labels created")
    
    # Alternative: Map from categorical codes (if present)
    if 'Activity_Code' in df.columns:
        activity_code_mapping = {
            '0': 'Inactive',
            '500': 'Low Activity',
            '1000': 'Moderate Activity',
            '1500': 'High Activity'
        }
        df['Activity_Status_From_Code'] = df['Activity_Code'].map(activity_code_mapping)
    
    # Map mood labels
    mood_mapping = {
        1: 'Very Negative',
        2: 'Negative',
        3: 'Neutral',
        4: 'Positive',
        5: 'Very Positive'
    }
    df['Mood_Label'] = df['Mood'].map(mood_mapping)
    
    # Binary mood variable
    df['Positive_Mood'] = (df['Mood'] >= 4).astype(int)
    
    # 4000-step threshold variable
    df['Above_4000_Steps'] = (df['Steps'] >= 4000).astype(int)
    
    # Day of week features
    if 'Date' in df.columns:
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    print("✓ Mood labels and binary features created")
    print("✓ Feature engineering complete")
    
    return df

# ============================================================================
# PART 5: EXPLORATORY DATA ANALYSIS
# ============================================================================

def display_summary(df):
    """Display comprehensive summary statistics"""
    print("\n" + "=" * 70)
    print("DATA SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nDataset Overview:")
    print(f"  Total Records: {len(df):,}")
    print(f"  Unique Users: {df['User_ID'].nunique()}")
    if 'Date' in df.columns:
        print(f"  Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    print(f"\nActivity Metrics:")
    print(f"  Average Steps: {df['Steps'].mean():.0f}")
    print(f"  Median Steps: {df['Steps'].median():.0f}")
    print(f"  Max Steps: {df['Steps'].max()}")
    
    if 'Sleep_Hours' in df.columns:
        print(f"  Average Sleep: {df['Sleep_Hours'].mean():.2f} hours")
    
    if 'Calories' in df.columns:
        print(f"  Average Calories: {df['Calories'].mean():.0f}")
    
    print(f"\nMood Distribution:")
    mood_dist = df['Mood_Label'].value_counts().sort_index()
    for mood, count in mood_dist.items():
        pct = count / len(df) * 100
        print(f"  {mood:15s}: {count:5,} ({pct:5.1f}%)")
    
    print(f"\nActivity Status Distribution:")
    activity_dist = df['Activity_Status'].value_counts()
    for status, count in activity_dist.items():
        pct = count / len(df) * 100
        print(f"  {status:20s}: {count:5,} ({pct:5.1f}%)")
    
    positive_mood_rate = df['Positive_Mood'].mean() * 100
    print(f"\nPositive Mood Rate: {positive_mood_rate:.1f}%")

# ============================================================================
# PART 6: CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations(df):
    """
    Analyze correlations between activity and mood
    """
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: STEPS vs MOOD")
    print("=" * 70)
    
    # Compute correlation
    correlation, p_value = pearsonr(df['Steps'], df['Mood'])
    
    print(f"\nPearson Correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significance: {'SIGNIFICANT (p<0.05)' if p_value < 0.05 else 'Not significant'}")
    
    if correlation > 0:
        print(f"Interpretation: Positive correlation - higher steps associated with better mood")
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['Steps'], df['Mood'], alpha=0.4, 
                         c=df['Mood'], cmap='RdYlGn', s=30, edgecolors='none')
    
    # Add trend line
    z = np.polyfit(df['Steps'], df['Mood'], 1)
    p = np.poly1d(z)
    plt.plot(df['Steps'].sort_values(), p(df['Steps'].sort_values()), 
             "r--", linewidth=2, alpha=0.8, label=f'Trend (r={correlation:.3f})')
    
    plt.xlabel('Steps', fontsize=12, fontweight='bold')
    plt.ylabel('Mood Score', fontsize=12, fontweight='bold')
    plt.title('Steps vs Mood Correlation Analysis', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Mood Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/claude/steps_mood_correlation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Correlation plot saved: steps_mood_correlation.png")
    plt.show()
    
    # Correlation matrix
    corr_cols = ['Steps', 'Mood']
    if 'Calories' in df.columns:
        corr_cols.append('Calories')
    if 'Sleep_Hours' in df.columns:
        corr_cols.append('Sleep_Hours')
    
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, fmt='.3f')
    plt.title('Correlation Matrix: Activity & Well-being Metrics', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation matrix saved: correlation_matrix.png")
    plt.show()
    
    return correlation, p_value

# ============================================================================
# PART 7: ACTIVITY-MOOD PATTERN ANALYSIS
# ============================================================================

def analyze_activity_mood_patterns(df):
    """
    Analyze mood patterns by activity level
    """
    print("\n" + "=" * 70)
    print("MOOD PATTERNS BY ACTIVITY LEVEL")
    print("=" * 70)
    
    # Group by activity status
    mood_by_activity = df.groupby('Activity_Status')['Mood'].agg([
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Count', 'count')
    ]).round(2)
    
    print("\nMood Statistics by Activity Level:")
    print(mood_by_activity.sort_values('Mean', ascending=False))
    
    # Positive mood rate by activity
    positive_by_activity = df.groupby('Activity_Status')['Positive_Mood'].mean() * 100
    print("\nPositive Mood Rate (%) by Activity:")
    print(positive_by_activity.sort_values(ascending=False).round(1))
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    activity_order = ['Inactive', 'Moderately Active', 'Active', 'Highly Active']
    sns.boxplot(data=df, x='Activity_Status', y='Mood', order=activity_order, 
                palette='Set2', ax=ax1)
    ax1.set_xlabel('Activity Status', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mood Score', fontsize=12, fontweight='bold')
    ax1.set_title('Mood Distribution by Activity Level', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Bar plot - positive mood rate
    positive_ordered = positive_by_activity.reindex(activity_order)
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    bars = ax2.bar(range(len(activity_order)), positive_ordered.values, 
                   color=colors, alpha=0.7, edgecolor='navy')
    ax2.set_xticks(range(len(activity_order)))
    ax2.set_xticklabels(activity_order, rotation=15)
    ax2.set_xlabel('Activity Status', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Positive Mood Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Positive Mood Likelihood by Activity', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(positive_ordered.values):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/claude/mood_by_activity.png', dpi=300, bbox_inches='tight')
    print("\n✓ Activity-mood analysis saved: mood_by_activity.png")
    plt.show()

# ============================================================================
# PART 8: 4000-STEP THRESHOLD ANALYSIS
# ============================================================================

def analyze_4000_threshold(df):
    """
    Analyze the critical 4000-step threshold finding
    """
    print("\n" + "=" * 70)
    print("4000-STEP THRESHOLD ANALYSIS")
    print("=" * 70)
    
    # Split data
    above_4000 = df[df['Above_4000_Steps'] == 1]
    below_4000 = df[df['Above_4000_Steps'] == 0]
    
    # Mood comparison
    above_mood = above_4000['Mood'].mean()
    below_mood = below_4000['Mood'].mean()
    improvement = ((above_mood - below_mood) / below_mood) * 100
    
    print(f"\nMood Comparison:")
    print(f"  Average Mood (≥4000 steps): {above_mood:.2f}")
    print(f"  Average Mood (<4000 steps): {below_mood:.2f}")
    print(f"  Mood Improvement: {improvement:+.1f}%")
    
    # Positive mood likelihood
    above_positive = above_4000['Positive_Mood'].mean() * 100
    below_positive = below_4000['Positive_Mood'].mean() * 100
    diff = above_positive - below_positive
    
    print(f"\nPositive Mood Likelihood:")
    print(f"  Rate (≥4000 steps): {above_positive:.1f}%")
    print(f"  Rate (<4000 steps): {below_positive:.1f}%")
    print(f"  Increased Likelihood: {diff:+.1f} percentage points")
    
    # Statistical test
    contingency = pd.crosstab(df['Above_4000_Steps'], df['Positive_Mood'])
    chi2, p_val, dof, expected = chi2contingency(contingency)
    
    print(f"\nStatistical Significance:")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  P-value: {p_val:.6f}")
    print(f"  Result: {'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'Not significant'}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mood comparison
    categories = ['<4000 Steps', '≥4000 Steps']
    mood_values = [below_mood, above_mood]
    bars1 = ax1.bar(categories, mood_values, color=['coral', 'lightseagreen'], 
                    alpha=0.7, edgecolor='navy', linewidth=2)
    ax1.set_ylabel('Average Mood Score', fontsize=11, fontweight='bold')
    ax1.set_title('Key Finding: Mood Improvement Above 4000 Steps', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 5)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(mood_values):
        ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    ax1.annotate(f'+{improvement:.1f}%', xy=(0.5, max(mood_values)), 
                xytext=(0.5, max(mood_values) + 0.3),
                ha='center', fontsize=12, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Positive mood rate
    positive_values = [below_positive, above_positive]
    bars2 = ax2.bar(categories, positive_values, color=['coral', 'lightseagreen'],
                    alpha=0.7, edgecolor='navy', linewidth=2)
    ax2.set_ylabel('Positive Mood Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Positive Mood Likelihood: Critical Threshold', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(positive_values):
        ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/claude/4000_step_threshold.png', dpi=300, bbox_inches='tight')
    print("\n✓ Threshold analysis saved: 4000_step_threshold.png")
    plt.show()
    
    return {
        'above_mood': above_mood,
        'below_mood': below_mood,
        'improvement_pct': improvement,
        'above_positive': above_positive,
        'below_positive': below_positive,
        'chi2_pvalue': p_val
    }

# ============================================================================
# PART 9: USER SEGMENTATION
# ============================================================================

def segment_users(df):
    """
    Segment users based on steps and mood for targeted engagement
    """
    print("\n" + "=" * 70)
    print("USER SEGMENTATION ANALYSIS")
    print("=" * 70)
    
    def assign_segment(row):
        """Assign user to engagement segment"""
        steps = row['Steps']
        mood = row['Mood']
        
        if steps >= 10000 and mood >= 4:
            return 'Champions'
        elif steps >= 10000 and mood < 4:
            return 'Active but Struggling'
        elif steps < 10000 and mood >= 4:
            return 'Happy but Inactive'
        else:
            return 'At Risk'
    
    df['User_Segment'] = df.apply(assign_segment, axis=1)
    
    # Segment distribution
    segment_counts = df['User_Segment'].value_counts()
    
    print("\nSegment Distribution:")
    for segment, count in segment_counts.items():
        pct = count / len(df) * 100
        print(f"  {segment:25s}: {count:5,} ({pct:5.1f}%)")
    
    # Segment characteristics
    print("\nSegment Characteristics:")
    segment_stats = df.groupby('User_Segment').agg({
        'Steps': ['mean', 'median'],
        'Mood': ['mean', 'median'],
        'Positive_Mood': lambda x: x.mean() * 100
    }).round(2)
    segment_stats.columns = ['Steps_Mean', 'Steps_Median', 'Mood_Mean', 
                             'Mood_Median', 'Positive_Mood_%']
    print(segment_stats)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot with segments
    colors_map = {
        'Champions': 'green',
        'Happy but Inactive': 'gold',
        'Active but Struggling': 'orange',
        'At Risk': 'red'
    }
    
    for segment in df['User_Segment'].unique():
        segment_data = df[df['User_Segment'] == segment]
        ax1.scatter(segment_data['Steps'], segment_data['Mood'],
                   label=segment, alpha=0.5, s=40,
                   color=colors_map.get(segment, 'gray'), edgecolors='black', linewidth=0.5)
    
    ax1.axvline(x=10000, color='black', linestyle='--', alpha=0.6, label='10k Steps')
    ax1.axhline(y=4, color='black', linestyle='--', alpha=0.6, label='Positive Mood')
    ax1.set_xlabel('Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mood Score', fontsize=12, fontweight='bold')
    ax1.set_title('User Segmentation: Steps vs Mood', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Segment size bar chart
    segment_order = ['Champions', 'Happy but Inactive', 'Active but Struggling', 'At Risk']
    segment_counts_ordered = segment_counts.reindex(segment_order)
    colors_ordered = [colors_map[s] for s in segment_order]
    
    bars = ax2.barh(range(len(segment_order)), segment_counts_ordered.values,
                    color=colors_ordered, alpha=0.7, edgecolor='navy', linewidth=1.5)
    ax2.set_yticks(range(len(segment_order)))
    ax2.set_yticklabels(segment_order)
    ax2.set_xlabel('Number of Records', fontsize=12, fontweight='bold')
    ax2.set_title('Segment Distribution', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(segment_counts_ordered.values):
        ax2.text(v + len(df)*0.01, i, f'{v:,}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/claude/user_segmentation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Segmentation analysis saved: user_segmentation.png")
    plt.show()
    
    return segment_counts

# ============================================================================
# PART 10: ENGAGEMENT STRATEGIES
# ============================================================================

def generate_engagement_strategies(segment_counts, total_records):
    """
    Generate personalized engagement strategies
    """
    print("\n" + "=" * 70)
    print("PERSONALIZED ENGAGEMENT STRATEGIES & NUDGES")
    print("=" * 70)
    
    strategies = {
        'Champions': {
            'priority': 'Retention',
            'nudge': 'Maintain momentum with milestone celebrations',
            'actions': [
                'Send weekly achievement badges and streaks',
                'Offer advanced challenges (15k+ step goals)',
                'Encourage community leadership (share tips)',
                'Provide exclusive content/rewards',
                'Celebrate monthly milestones'
            ],
            'expected_impact': 'Maintain 95%+ retention, increase social engagement'
        },
        'Happy but Inactive': {
            'priority': 'Activation',
            'nudge': 'Gentle encouragement to build activity habits',
            'actions': [
                'Set achievable micro-goals (start at 5000 steps)',
                'Send motivational reminders (timing: 2-4 PM)',
                'Suggest fun activities aligned with interests',
                'Share mood-activity connection insights',
                'Gamify with beginner challenges'
            ],
            'expected_impact': '25-40% increase in daily step count within 30 days'
        },
        'Active but Struggling': {
            'priority': 'Well-being Support',
            'nudge': 'Wellness check-ins and mental health support',
            'actions': [
                'Recommend mindfulness/meditation exercises',
                'Suggest social activities and group workouts',
                'Provide mental health resources',
                'Check for overtraining/burnout signs',
                'Encourage rest days and recovery'
            ],
            'expected_impact': 'Improve mood scores by 15-20%, reduce churn by 30%'
        },
        'At Risk': {
            'priority': 'Critical Intervention',
            'nudge': 'Immediate empathetic outreach and support',
            'actions': [
                'Send empathetic check-in messages (not pushy)',
                'Offer micro-goals (500-step increments)',
                'Provide mental health hotline information',
                'Connect with support groups/communities',
                'Suggest simple mood-boosting activities',
                'Offer personalized coaching sessions'
            ],
            'expected_impact': 'Reduce churn by 40%, re-engage 20-30% within 2 weeks'
        }
    }
    
    for segment, strategy in strategies.items():
        count = segment_counts.get(segment, 0)
        pct = (count / total_records * 100) if total_records > 0 else 0
        
        print(f"\n{'=' * 70}")
        print(f"{segment.upper()}")
        print(f"{'=' * 70}")
        print(f"Size: {count:,} records ({pct:.1f}% of total)")
        print(f"Priority: {strategy['priority']}")
        print(f"\nNudge Strategy: {strategy['nudge']}")
        print(f"\nRecommended Actions:")
        for i, action in enumerate(strategy['actions'], 1):
            print(f"  {i}. {action}")
        print(f"\nExpected Impact: {strategy['expected_impact']}")
    
    # Create strategy summary
    return strategies

# ============================================================================
# PART 11: EXPORT RESULTS
# ============================================================================

def export_results(df, correlation, threshold_stats, segment_counts, strategies):
    """
    Export all analysis results
    """
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)
    
    # Summary statistics
    summary = {
        'Total_Records': [len(df)],
        'Unique_Users': [df['User_ID'].nunique()],
        'Avg_Steps': [df['Steps'].mean()],
        'Avg_Mood': [df['Mood'].mean()],
        'Steps_Mood_Correlation': [correlation],
        'Positive_Mood_Rate': [df['Positive_Mood'].mean() * 100],
        'Above_4000_Steps_Pct': [(df['Steps'] >= 4000).mean() * 100],
        'Mood_Improvement_Above_4000': [threshold_stats['improvement_pct']]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('/home/claude/fitbit_analysis_summary.csv', index=False)
    print("✓ Summary statistics: fitbit_analysis_summary.csv")
    
    # Segmented users
    output_cols = ['User_ID', 'Date', 'Steps', 'Mood', 'Activity_Status', 
                   'Mood_Label', 'User_Segment']
    available_cols = [col for col in output_cols if col in df.columns]
    df[available_cols].to_csv('/home/claude/fitbit_segmented_users.csv', index=False)
    print("✓ Segmented users: fitbit_segmented_users.csv")
    
    # Engagement strategies
    strategy_records = []
    for segment, strategy in strategies.items():
        strategy_records.append({
            'Segment': segment,
            'User_Count': segment_counts.get(segment, 0),
            'Priority': strategy['priority'],
            'Nudge': strategy['nudge'],
            'Expected_Impact': strategy['expected_impact']
        })
    
    strategy_df = pd.DataFrame(strategy_records)
    strategy_df.to_csv('/home/claude/engagement_strategies.csv', index=False)
    print("✓ Engagement strategies: engagement_strategies.csv")
    
    # Threshold analysis
    threshold_df = pd.DataFrame([threshold_stats])
    threshold_df.to_csv('/home/claude/4000_threshold_analysis.csv', index=False)
    print("✓ Threshold analysis: 4000_threshold_analysis.csv")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("\n" + "=" * 70)
    print("FITBIT BEHAVIORAL ANALYTICS - COMPLETE ANALYSIS")
    print("=" * 70)
    
    # Step 1: Load data
    df = load_data()  # Use load_data('your_file.csv') for real data
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Step 3: Engineer features
    df = engineer_features(df)
    
    # Step 4: Display summary
    display_summary(df)
    
    # Step 5: Correlation analysis
    correlation, p_value = analyze_correlations(df)
    
    # Step 6: Activity-mood patterns
    analyze_activity_mood_patterns(df)
    
    # Step 7: 4000-step threshold analysis
    threshold_stats = analyze_4000_threshold(df)
    
    # Step 8: User segmentation
    segment_counts = segment_users(df)
    
    # Step 9: Engagement strategies
    strategies = generate_engagement_strategies(segment_counts, len(df))
    
    # Step 10: Export results
    export_results(df, correlation, threshold_stats, segment_counts, strategies)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  📊 steps_mood_correlation.png")
    print("  📊 correlation_matrix.png")
    print("  📊 mood_by_activity.png")
    print("  📊 4000_step_threshold.png")
    print("  📊 user_segmentation.png")
    print("  📄 fitbit_analysis_summary.csv")
    print("  📄 fitbit_segmented_users.csv")
    print("  📄 engagement_strategies.csv")
    print("  📄 4000_threshold_analysis.csv")
    print("\n✓ All analysis complete and files saved successfully!")

if __name__ == "__main__":
    main()
