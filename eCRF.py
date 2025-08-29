import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color scheme configuration
COLORS = {
    'good': '#004a3a',      # Emerald
    'bad': '#9b111e',       # Ruby Red
    'warning': '#ff9f14',   # Light Orange
    'neutral': '#708090'    # Slate Gray
}

def get_rag_color(value, thresholds=(0.3, 0.5), reverse=False):
    """Return RAG colors based on value and thresholds"""
    if reverse:  # For completion time (lower is better)
        if value <= thresholds[0]:
            return COLORS['good']
        elif value <= thresholds[1]:
            return COLORS['warning']
        else:
            return COLORS['bad']
    else:  # For late rate (lower is better)
        if value <= thresholds[0]:
            return COLORS['good']
        elif value <= thresholds[1]:
            return COLORS['warning']
        else:
            return COLORS['bad']

def validate_data(df):
    """Validate data quality and report issues"""
    print("="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    issues = {}
    
    # Check required columns
    required_columns = [
        '# Forms Complete > 14 days',
        '# Forms Complete ≤ 14 days',
        'Country', 'Form Name', 'Study Phase'
    ]
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        issues['Missing Columns'] = missing
        print(f"❌ Missing required columns: {missing}")
    else:
        print("✅ All required columns present")
    
    # Check for negative values
    for col in ['# Forms Complete > 14 days', '# Forms Complete ≤ 14 days']:
        if col in df.columns and (df[col] < 0).any():
            issues[f'Negative values in {col}'] = df[df[col] < 0].shape[0]
            print(f"⚠️  Found {df[df[col] < 0].shape[0]} negative values in {col}")
    
    # Check completion times
    if 'AVG Form Complete CT' in df.columns:
        df['AVG_Form_Complete_CT_Clean'] = pd.to_numeric(df['AVG Form Complete CT'], errors='coerce')
        unrealistic = df[df['AVG_Form_Complete_CT_Clean'] > 365].shape[0]
        if unrealistic > 0:
            print(f"⚠️  Found {unrealistic} records with completion time > 365 days")
    
    # Check for missing critical data
    missing_countries = df['Country'].isna().sum()
    if missing_countries > 0:
        print(f"⚠️  Found {missing_countries} records with missing Country")
    
    print(f"✅ Data validation complete. Total issues found: {len(issues)}")
    return issues

# Load the data
df = pd.read_csv(r"C:\Users\a239584\Downloads\Coding\eCRF_data\Data Entry compliance 14May2025 all dimentions.csv")

# Display basic info about the dataset
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# Validate data
validation_issues = validate_data(df)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics on late forms
print("\nLate Forms Analysis:")
print(f"Total records: {len(df)}")
print(f"Records with forms completed > 14 days: {df['# Forms Complete > 14 days'].sum()}")
print(f"Records with forms completed ≤ 14 days: {df['# Forms Complete ≤ 14 days'].sum()}")
print(f"Percentage of late forms: {(df['# Forms Complete > 14 days'].sum() / (df['# Forms Complete > 14 days'].sum() + df['# Forms Complete ≤ 14 days'].sum()) * 100):.2f}%")


# Create a comprehensive eCRF Late Submission Analysis Script

def analyze_ecrf_late_submissions(df):
    """
    Comprehensive analysis of late eCRF submissions with trend identification
    """
  
    # Create a binary late indicator
    df['Is_Late'] = df['# Forms Complete > 14 days'] > 0
    df['Late_Rate'] = df['# Forms Complete > 14 days'] / (df['# Forms Complete ≤ 14 days'] + df['# Forms Complete > 14 days'])
  
    # Convert AVG Form Complete CT to numeric, handling '-' values
    df['AVG_Form_Complete_CT_Clean'] = pd.to_numeric(df['AVG Form Complete CT'], errors='coerce')
  
    results = {}
  
    # 1. COUNTRY/REGION ANALYSIS
    print("="*60)
    print("1. COUNTRY/REGION LATE SUBMISSION ANALYSIS")
    print("="*60)
    
    country_analysis = df.groupby('Country').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).round(3)
    
    country_analysis.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    country_analysis = country_analysis.sort_values('Late_Rate', ascending=False)
    
    print("Country Performance (sorted by late rate):")
    print(country_analysis)
    results['country_analysis'] = country_analysis
    
    # Region analysis
    if 'Region' in df.columns:
        region_analysis = df.groupby('Region').agg({
            'Is_Late': ['count', 'sum', 'mean'],
            'AVG_Form_Complete_CT_Clean': 'mean'
        }).round(3)
        
        region_analysis.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
        print(f"\nRegion Performance:")
        print(region_analysis)
        results['region_analysis'] = region_analysis
  
    # 2. FORM TYPE ANALYSIS
    print("\n" + "="*60)
    print("2. FORM TYPE LATE SUBMISSION ANALYSIS")
    print("="*60)
    
    form_analysis = df.groupby('Form Name').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).round(3)
    
    form_analysis.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    form_analysis = form_analysis.sort_values('Late_Rate', ascending=False)
    
    print("Top 10 Most Problematic Form Types:")
    print(form_analysis.head(10))
    results['form_analysis'] = form_analysis
  
    # 3. STUDY PHASE ANALYSIS
    print("\n" + "="*60)
    print("3. STUDY PHASE ANALYSIS")
    print("="*60)
    
    phase_analysis = df.groupby('Study Phase').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).round(3)
    
    phase_analysis.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    print(phase_analysis)
    results['phase_analysis'] = phase_analysis
  
    # 4. VISIT TYPE ANALYSIS
    print("\n" + "="*60)
    print("4. VISIT TYPE ANALYSIS")
    print("="*60)
    
    visit_analysis = df.groupby('Visit Name').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).round(3)
    
    visit_analysis.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    visit_analysis = visit_analysis.sort_values('Late_Rate', ascending=False)
    
    print("Top 10 Most Problematic Visit Types:")
    print(visit_analysis.head(10))
    results['visit_analysis'] = visit_analysis
  
    # 5. SITE PERFORMANCE ANALYSIS
    print("\n" + "="*60)
    print("5. SITE PERFORMANCE ANALYSIS")
    print("="*60)
    
    site_analysis = df.groupby(['Study Site', 'Site Name (CTMS)', 'Country']).agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).round(3)
    
    site_analysis.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    site_analysis = site_analysis.sort_values('Late_Rate', ascending=False)
    
    print("Site Performance (showing sites with >5 forms):")
    site_filtered = site_analysis[site_analysis['Total_Forms'] > 5]
    print(site_filtered)
    results['site_analysis'] = site_analysis
    
    return results

# Run the analysis
analysis_results = analyze_ecrf_late_submissions(df)

# Create an output folder
os.makedirs("plots", exist_ok=True)

# 1. Country Late Rate Analysis
plt.figure(figsize=(16,8))
country_data = analysis_results['country_analysis']
late_percentages = country_data['Late_Rate'] * 100  # Convert to percentage

# Apply RAG colors
colors = [get_rag_color(x/100) for x in late_percentages]

bars = plt.bar(country_data.index, late_percentages, color=colors)
plt.title('Late Submission Rate by Country', fontsize=14, fontweight='bold')
plt.xlabel('Countries')
plt.ylabel('Percentage of Late Submissions (%)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, max(late_percentages) * 1.1)

# Add exact number of late submissions on top of bars
for i, (bar, late_count) in enumerate(zip(bars, country_data['Late_Forms'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(late_count)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/1_country_late_rate.png", dpi=300)
plt.close()


# 2. Form Type Analysis (Top 10 most problematic)
plt.figure(figsize=(12,8))
form_data = analysis_results['form_analysis']
top_10_forms = form_data.head(10)
other_forms = form_data.iloc[10:]

# Prepare data for pie chart
pie_data = top_10_forms['Late_Forms'].tolist()
pie_labels = [label[:30] + '...' if len(label) > 30 else label for label in top_10_forms.index.tolist()]

if len(other_forms) > 0:
    pie_data.append(other_forms['Late_Forms'].sum())
    pie_labels.append('Other')

# Create pie chart
colors_pie = [COLORS['bad']] * len(pie_data)  # All red as requested
wedges, texts, autotexts = plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                  colors=colors_pie, startangle=90)

# Add counts to the labels
for i, (wedge, count) in enumerate(zip(wedges, pie_data)):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = wedge.r * 0.7 * np.cos(np.radians(angle))
    y = wedge.r * 0.7 * np.sin(np.radians(angle))
    plt.text(x, y, f'n={int(count)}', ha='center', va='center', fontweight='bold', fontsize=8)

plt.title('Top 10 Most Problematic Form Types', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig("plots/2_form_type_analysis.png", dpi=300)
plt.close()


# 3. Study Phase Analysis
plt.figure(figsize=(10,6))
phase_data = analysis_results['phase_analysis']
phase_percentages = phase_data['Late_Rate'] * 100

colors = [get_rag_color(x/100) for x in phase_percentages]
bars = plt.bar(phase_data.index, phase_percentages, color=colors)

plt.title('Late Submission Rate by Study Phase', fontsize=14, fontweight='bold')
plt.xlabel('Study Phase')
plt.ylabel('Percentage of Late Submissions (%)')
plt.xticks(rotation=45)

# Add number of late submissions on top of bars
for bar, late_count in zip(bars, phase_data['Late_Forms']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(late_count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/3_study_phase_analysis.png", dpi=300)
plt.close()


# 4. Site Performance Analysis
plt.figure(figsize=(16,8))
site_data = analysis_results['site_analysis'][analysis_results['site_analysis']['Total_Forms'] > 5]
site_data = site_data.sort_values('Late_Rate', ascending=False).head(20)
site_percentages = site_data['Late_Rate'] * 100

colors = [get_rag_color(x/100) for x in site_percentages]
bars = plt.bar(range(len(site_data)), site_percentages, color=colors)

plt.title('Site Performance (Top 20 Worst, >5 forms)', fontsize=14, fontweight='bold')
plt.xlabel('Sites')
plt.ylabel('Percentage of Late Submissions (%)')
plt.xticks(range(len(site_data)),
           [f"{idx[0]}-{idx[2]}" for idx in site_data.index],
           rotation=90, fontsize=8)

# Add number of late submissions on top of bars
for bar, late_count in zip(bars, site_data['Late_Forms']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(late_count)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/4_site_performance.png", dpi=300)
plt.close()


# 5. Average Completion Time by Country
plt.figure(figsize=(16,8))
country_time = analysis_results['country_analysis'].dropna(subset=['Avg_Completion_Time'])

colors = [get_rag_color(x, thresholds=(14, 30), reverse=True) for x in country_time['Avg_Completion_Time']]
bars = plt.bar(country_time.index, country_time['Avg_Completion_Time'], color=colors)

plt.title('Average Form Completion Time by Country', fontsize=14, fontweight='bold')
plt.xlabel('Countries')
plt.ylabel('Average Completion Time (days)')
plt.xticks(rotation=45, ha='right')

# Add 14-day reference line
plt.axhline(y=14, color=COLORS['bad'], linestyle='--', linewidth=2, 
           label='14-day threshold', alpha=0.8)
plt.legend()

plt.tight_layout()
plt.savefig("plots/5_avg_completion_time_by_country.png", dpi=300)
plt.close()


# 6. Therapy Area Analysis
plt.figure(figsize=(12,6))
therapy_analysis = df.groupby('Therapy Area').agg({
    'Is_Late': ['count', 'sum', 'mean'],
    'AVG_Form_Complete_CT_Clean': 'mean'
}).round(3)
therapy_analysis.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
therapy_analysis = therapy_analysis.sort_values('Late_Rate', ascending=False)

# Top 10 + Other
top_10_therapy = therapy_analysis.head(10)
other_therapy = therapy_analysis.iloc[10:]

therapy_data = top_10_therapy.copy()
if len(other_therapy) > 0:
    other_row = pd.DataFrame({
        'Total_Forms': [other_therapy['Total_Forms'].sum()],
        'Late_Forms': [other_therapy['Late_Forms'].sum()],
        'Late_Rate': [other_therapy['Late_Forms'].sum() / other_therapy['Total_Forms'].sum()],
        'Avg_Completion_Time': [other_therapy['Avg_Completion_Time'].mean()]
    }, index=['Other'])
    therapy_data = pd.concat([therapy_data, other_row])

therapy_percentages = therapy_data['Late_Rate'] * 100
bars = plt.bar(therapy_data.index, therapy_percentages, color=COLORS['bad'])

plt.title('Percentage of Late Submissions by Therapy Area (Top 10 + Other)', fontsize=14, fontweight='bold')
plt.xlabel('Therapy Area')
plt.ylabel('Percentage of Late Submissions (%)')
plt.xticks(rotation=45, ha='right')

# Add number of late submissions on top of bars
for bar, late_count in zip(bars, therapy_data['Late_Forms']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(late_count)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("plots/6_therapy_area_analysis.png", dpi=300)
plt.close()


# 7. Volume vs Late Rate Scatter (Sites)
plt.figure(figsize=(12,8))
site_scatter = analysis_results['site_analysis']

# Calculate medians for quadrant lines
median_volume = site_scatter['Total_Forms'].median()
median_late_rate = site_scatter['Late_Rate'].median()

# Create quadrants
colors_quad = []
labels_quad = []
for _, row in site_scatter.iterrows():
    volume = row['Total_Forms']
    late_rate = row['Late_Rate']
    
    if volume >= median_volume and late_rate >= median_late_rate:
        colors_quad.append(COLORS['bad'])  # High Volume, High Risk
        labels_quad.append('High Vol, High Risk')
    elif volume >= median_volume and late_rate < median_late_rate:
        colors_quad.append(COLORS['good'])  # High Volume, Low Risk
        labels_quad.append('High Vol, Low Risk')
    elif volume < median_volume and late_rate >= median_late_rate:
        colors_quad.append(COLORS['warning'])  # Low Volume, High Risk
        labels_quad.append('Low Vol, High Risk')
    else:
        colors_quad.append(COLORS['neutral'])  # Low Volume, Low Risk
        labels_quad.append('Low Vol, Low Risk')

scatter = plt.scatter(site_scatter['Total_Forms'], site_scatter['Late_Rate'] * 100,
                     c=colors_quad, alpha=0.6, s=60)

plt.title('Site Performance Quadrant Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Total Forms (Volume)')
plt.ylabel('Late Rate (%)')

# Add quadrant lines
plt.axhline(y=median_late_rate * 100, color='gray', linestyle=':', alpha=0.7, 
           label=f'Median Late Rate ({median_late_rate*100:.1f}%)')
plt.axvline(x=median_volume, color='gray', linestyle=':', alpha=0.7, 
           label=f'Median Volume ({median_volume:.0f})')

# Add quadrant labels
plt.text(median_volume * 1.5, median_late_rate * 100 * 1.2, 'High Vol\nHigh Risk', 
         ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['bad'], alpha=0.3))
plt.text(median_volume * 1.5, median_late_rate * 100 * 0.3, 'High Vol\nLow Risk', 
         ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['good'], alpha=0.3))
plt.text(median_volume * 0.3, median_late_rate * 100 * 1.2, 'Low Vol\nHigh Risk', 
         ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['warning'], alpha=0.3))
plt.text(median_volume * 0.3, median_late_rate * 100 * 0.3, 'Low Vol\nLow Risk', 
         ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['neutral'], alpha=0.3))

plt.legend()
plt.tight_layout()
plt.savefig("plots/7_site_volume_vs_late_rate.png", dpi=300)
plt.close()


# 8. Completion Time Distribution
plt.figure(figsize=(10,6))
completion_times = df['AVG_Form_Complete_CT_Clean'].dropna()
completion_times_filtered = completion_times[(completion_times >= 0) & (completion_times <= 200)]

# Create KDE plot
from scipy.stats import gaussian_kde
kde = gaussian_kde(completion_times_filtered)
x_range = np.linspace(0, 200, 1000)
kde_values = kde(x_range)

plt.fill_between(x_range, kde_values, alpha=0.7, color=COLORS['neutral'])
plt.plot(x_range, kde_values, color='black', linewidth=2)

# Add 14-day reference line
plt.axvline(x=14, color=COLORS['bad'], linestyle='--', linewidth=2, 
           label='14-day threshold')

# Add median and quartiles
median_time = completion_times_filtered.median()
q25 = completion_times_filtered.quantile(0.25)
q75 = completion_times_filtered.quantile(0.75)

plt.axvline(x=median_time, color=COLORS['good'], linestyle='-', linewidth=2, 
           label=f'Median ({median_time:.1f} days)')
plt.axvline(x=q25, color=COLORS['warning'], linestyle=':', alpha=0.7, 
           label=f'Q1 ({q25:.1f} days)')
plt.axvline(x=q75, color=COLORS['warning'], linestyle=':', alpha=0.7, 
           label=f'Q3 ({q75:.1f} days)')

plt.title('Distribution of Form Completion Times (Global)', fontsize=14, fontweight='bold')
plt.xlabel('Completion Time (days)')
plt.ylabel('Density')
plt.xlim(0, 200)
plt.legend()
plt.tight_layout()
plt.savefig("plots/8_completion_time_distribution.png", dpi=300)
plt.close()

# 9. Trend Over Time (Global, Country, Site)
plt.figure(figsize=(14,8))
df['Completion_Month'] = pd.to_datetime(df['Form Complete Date'], errors='coerce').dt.to_period('M')
trend = df.groupby('Completion_Month')['Is_Late'].mean().reset_index()
trend['On_Time'] = 1 - trend['Is_Late']

# Convert period to string for better x-axis labels
trend['Month_Str'] = trend['Completion_Month'].astype(str)

# Plot global trend
plt.plot(range(len(trend)), trend['On_Time']*100, marker='o', linewidth=3, 
         label="Global", markersize=8, color=COLORS['neutral'])

# Add 5 worst performing countries
worst_countries = analysis_results['country_analysis'].head(5).index
colors_trend = [COLORS['bad'], COLORS['warning'], '#8B0000', '#FF4500', '#DC143C']
for i, country in enumerate(worst_countries):
    country_trend = df[df['Country']==country].groupby('Completion_Month')['Is_Late'].mean().reset_index()
    if len(country_trend) > 0:
        country_trend['On_Time'] = 1 - country_trend['Is_Late']
        plt.plot(range(len(country_trend)), country_trend['On_Time']*100, 
                marker='s', label=f'{country} (worst)', color=colors_trend[i], markersize=4)

plt.title("On-Time % Trend Over Time", fontsize=14, fontweight="bold")
plt.ylabel("% On-Time (<14 days)")
plt.xlabel("Month")
plt.xticks(range(len(trend)), trend['Month_Str'], rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/9_trend_over_time.png", dpi=300, bbox_inches='tight')
plt.close()

# 10. On-Time vs Late Split
plt.figure(figsize=(8,6))
on_time = (df['Is_Late']==False).sum()
late = (df['Is_Late']==True).sum()
not_complete = df['Form Complete Date'].isna().sum()
total = on_time + late + not_complete

categories = ['On-Time (≤14d)', 'Late (>14d)', 'Not Complete']
values = [on_time, late, not_complete]
percentages = [v/total*100 for v in values]

bars = plt.bar(categories, values, color=[COLORS['good'], COLORS['bad'], COLORS['neutral']])
plt.title("Overall On-Time vs Late Split", fontsize=14, fontweight="bold")
plt.ylabel("Number of Forms")

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, percentages)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{pct:.1f}%\n({values[i]:,})', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("plots/10_on_time_vs_late_split.png", dpi=300)
plt.close()

# 11. CRA Performance
cra_perf = df.groupby('CRA').agg({
    'Is_Late': ['count', 'mean'],
}).round(3)
cra_perf.columns = ['Total_Forms', 'Late_Rate']
cra_perf['On_Time_Pct'] = (1 - cra_perf['Late_Rate']) * 100
cra_perf = cra_perf[cra_perf['Total_Forms'] >= 5]  # Filter CRAs with at least 5 forms

# Best 10 CRAs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

best_cras = cra_perf.sort_values('On_Time_Pct', ascending=False).head(10)
bars1 = ax1.barh(best_cras.index, best_cras['On_Time_Pct'], color=COLORS['good'])
ax1.set_title("Top 10 Best CRA Performance", fontsize=14, fontweight="bold")
ax1.set_xlabel("% On-Time (<14 days)")

# Add form counts on bars
for i, (bar, forms) in enumerate(zip(bars1, best_cras['Total_Forms'])):
    width = bar.get_width()
    ax1.text(width + 1, bar.get_y() + bar.get_height()/2.,
             f'n={int(forms)}', ha='left', va='center', fontsize=8)

# Worst 10 CRAs
worst_cras = cra_perf.sort_values('On_Time_Pct', ascending=True).head(10)
bars2 = ax2.barh(worst_cras.index, worst_cras['On_Time_Pct'], color=COLORS['bad'])
ax2.set_title("Top 10 Worst CRA Performance", fontsize=14, fontweight="bold")
ax2.set_xlabel("% On-Time (<14 days)")

# Add form counts on bars
for i, (bar, forms) in enumerate(zip(bars2, worst_cras['Total_Forms'])):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height()/2.,
             f'n={int(forms)}', ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("plots/11_cra_performance.png", dpi=300)
plt.close()

# 12. Visit Type Analysis
plt.figure(figsize=(14,6))
visit_late_rate = df.groupby('Visit Name').agg({
    'Is_Late': ['count', 'sum', 'mean']
}).round(3)
visit_late_rate.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate']
visit_late_rate = visit_late_rate[visit_late_rate['Total_Forms'] >= 5]  # Filter visits with at least 5 forms
visit_late_rate = visit_late_rate.sort_values('Late_Rate', ascending=False).head(15)

visit_percentages = visit_late_rate['Late_Rate'] * 100
colors = [get_rag_color(x/100) for x in visit_percentages]

bars = plt.barh(visit_late_rate.index, visit_percentages, color=colors)
plt.title("Late Rate by Visit Type (Top 15)", fontsize=14, fontweight="bold")
plt.xlabel("Late Rate (%)")
plt.ylabel("Visit Type")

# Add percentages on bars
for bar, pct in zip(bars, visit_percentages):
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
             f'{pct:.1f}%', ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("plots/12_visit_type_analysis.png", dpi=300)
plt.close()

# 13. RAG Benchmark (Countries)
plt.figure(figsize=(16,8))
country_perf = df.groupby('Country')['Is_Late'].mean().reset_index()
country_perf['On_Time_Pct'] = (1 - country_perf['Is_Late']) * 100
country_perf = country_perf.sort_values('On_Time_Pct', ascending=False)  # Descending order

colors = country_perf['On_Time_Pct'].apply(
    lambda x: COLORS['good'] if x >= 80 else (COLORS['warning'] if x >= 60 else COLORS['bad'])
)

bars = plt.bar(country_perf['Country'], country_perf['On_Time_Pct'], color=colors)
plt.title("RAG Benchmark: On-Time Submissions by Country (Descending Order)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha='right')
plt.ylabel("% On-Time (<14 days)")
plt.axhline(y=80, color=COLORS['good'], linestyle='--', alpha=0.7, label='80% threshold (Good)')
plt.axhline(y=60, color=COLORS['warning'], linestyle='--', alpha=0.7, label='60% threshold (Warning)')
plt.legend()
plt.tight_layout()
plt.savefig("plots/13_rag_benchmark.png", dpi=300)
plt.close()

# 14. Best vs Worst Sites Leaderboard
plt.figure(figsize=(14,10))
site_perf = df.groupby(['Study Site', 'Country']).agg({
    'Is_Late': ['count', 'sum', 'mean']
}).round(3)
site_perf.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate']
site_perf['On_Time_Pct'] = (1 - site_perf['Late_Rate']) * 100
site_perf = site_perf[site_perf['Total_Forms'] >= 5]  # Filter sites with at least 5 forms

best = site_perf.sort_values('On_Time_Pct', ascending=False).head(10)
worst = site_perf.sort_values('On_Time_Pct', ascending=True).head(10)
leaderboard = pd.concat([best, worst])

# Create labels with site and country
leaderboard['Site_Country'] = [f"{idx[0]} ({idx[1]})" for idx in leaderboard.index]

colors_leader = [COLORS['good']] * 10 + [COLORS['bad']] * 10
bars = plt.barh(leaderboard['Site_Country'], leaderboard['On_Time_Pct'], color=colors_leader)

plt.title("Leaderboard: Best & Worst Performing Sites", fontsize=14, fontweight="bold")
plt.xlabel("% On-Time")

# Add form counts on bars
for bar, forms in zip(bars, leaderboard['Total_Forms']):
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2.,
             f'n={int(forms)}', ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("plots/14_best_worst_leaderboard.png", dpi=300)
plt.close()

# 15. Early Warning Dashboard (simple rule)
plt.figure(figsize=(14,10))
site_perf_detailed = df.groupby(['Study Site', 'Country']).agg({
    'Is_Late': ['count', 'sum', 'mean']
}).round(3)
site_perf_detailed.columns = ['Total_Forms', 'Late_Forms', 'Late_Rate']
site_perf_detailed['On_Time_Pct'] = (1 - site_perf_detailed['Late_Rate']) * 100

# Calculate contribution to global late forms
total_late_forms = df['Is_Late'].sum()
site_perf_detailed['Late_Contribution_Pct'] = (site_perf_detailed['Late_Forms'] / total_late_forms) * 100

# Filter high risk sites and get top 25
risk_sites = site_perf_detailed[site_perf_detailed['On_Time_Pct'] < 50].sort_values('On_Time_Pct').head(25)

if len(risk_sites) > 0:
    risk_sites['Site_Country'] = [f"{idx[0]} ({idx[1]})" for idx in risk_sites.index]
    
    bars = plt.barh(risk_sites['Site_Country'], risk_sites['On_Time_Pct'], color=COLORS['bad'])
    plt.title("Early Warning: Top 25 High Risk Sites (<50% On-Time)", fontsize=14, fontweight="bold")
    plt.xlabel("% On-Time")
    
    # Add contribution percentages on bars
    for bar, contrib in zip(bars, risk_sites['Late_Contribution_Pct']):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2.,
                 f'{contrib:.1f}% of late forms', ha='left', va='center', fontsize=8)
else:
    plt.text(0.5, 0.5, 'No sites with <50% On-Time performance', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    plt.title("Early Warning: High Risk Sites (<50% On-Time)", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("plots/15_early_warning.png", dpi=300)
plt.close()

# LOGISTIC REGRESSION ANALYSIS
print("\n" + "="*80)
print("LOGISTIC REGRESSION ANALYSIS")
print("="*80)

# Prepare data for logistic regression
df_model = df.dropna(subset=['Country', 'Study Phase', 'Form Name', 'Visit Name'])

# Encode categorical variables
le_country = LabelEncoder()
le_phase = LabelEncoder()
le_form = LabelEncoder()
le_visit = LabelEncoder()

df_model['Country_encoded'] = le_country.fit_transform(df_model['Country'])
df_model['Phase_encoded'] = le_phase.fit_transform(df_model['Study Phase'])
df_model['Form_encoded'] = le_form.fit_transform(df_model['Form Name'])
df_model['Visit_encoded'] = le_visit.fit_transform(df_model['Visit Name'])

# Prepare features and target
X = df_model[['Country_encoded', 'Phase_encoded', 'Form_encoded', 'Visit_encoded']]
y = df_model['Is_Late']

# Fit logistic regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X, y)

# Get feature importance (coefficients)
feature_names = ['Country', 'Study Phase', 'Form Name', 'Visit Name']
coefficients = log_reg.coef_[0]

print("Logistic Regression Results:")
print("Feature Importance (Coefficients):")
for name, coef in zip(feature_names, coefficients):
    print(f"  {name}: {coef:.4f}")

# Model performance
from sklearn.metrics import classification_report, roc_auc_score
y_pred = log_reg.predict(X)
y_pred_proba = log_reg.predict_proba(X)[:, 1]

print(f"\nModel Performance:")
print(f"AUC Score: {roc_auc_score(y, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Summary Statistics
print("\n" + "="*80)
print("KEY INSIGHTS SUMMARY")
print("="*80)

print(f"OVERALL PERFORMANCE:")
print(f"   • Total forms analyzed: {len(df)}")
print(f"   • Late submission rate: {(df['Is_Late'].sum() / len(df) * 100):.1f}%")
print(f"   • Average completion time: {df['AVG_Form_Complete_CT_Clean'].mean():.1f} days")

print(f"\nHIGHEST RISK COUNTRIES:")
top_risk_countries = analysis_results['country_analysis'].head(3)
for country, data in top_risk_countries.iterrows():
    print(f"   • {country}: {data['Late_Rate']:.1%} late rate ({data['Late_Forms']:.0f}/{data['Total_Forms']:.0f} forms)")

print(f"\n SITE PERFORMANCE ISSUES:")
problematic_sites = analysis_results['site_analysis'][analysis_results['site_analysis']['Late_Rate'] > 0.7]
for (site_id, site_name, country), data in problematic_sites.iterrows():
    print(f"   • {site_id} ({country}): {data['Late_Rate']:.1%} late rate "
          f"({data['Late_Forms']:.0f}/{data['Total_Forms']:.0f} forms)")

print(f"\n📋 MOST PROBLEMATIC FORM TYPES:")
problem_forms = analysis_results['form_analysis'][analysis_results['form_analysis']['Late_Rate'] == 1.0].head(5)
for form_name, data in problem_forms.iterrows():
    print(f"   • {form_name[:50]}{'...' if len(form_name) > 50 else ''}: 100% late rate")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - All plots saved to 'plots/' directory")
print("="*80)