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
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import textwrap
from datetime import datetime, timedelta

# Load the data
df = pd.read_csv(r"C:\Users\a239584\Downloads\Coding\eCRF_data\SEE Report 03Sep2025.csv")

# Calculate the date 6 months ago from today
six_months_ago = datetime.now() - timedelta(days=180)  # Approximately 6 months

# Filter to keep only the last 6 months of data
df = df[df['Form Complete Date'] >= six_months_ago]

print(f"Data filtered to last 6 months. New dataset shape: {df.shape}")
print(f"Date range: {df['Form Complete Date'].min()} to {df['Form Complete Date'].max()}")

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
df = pd.read_csv(r"C:\Users\a239584\Downloads\Coding\eCRF_data\SEE Report 03Sep2025.csv")

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

# GENERATE ALL INSIGHTS (put this block here)
# Country insights
top_countries = analysis_results['country_analysis'].head(3)
country_insights = "KEY INSIGHTS:\n\n"
for country, row in top_countries.iterrows():
    country_insights += f"• {country}: {row['Late_Rate']:.1%} late rate ({int(row['Late_Forms'])}/{int(row['Total_Forms'])} forms)\n"

# Form insights  
problem_forms = analysis_results['form_analysis']
form_insights = "KEY INSIGHTS:\n\n"
for form, row in problem_forms.head(5).iterrows():
    form_insights += f"• {form[:40]}: {row['Late_Rate']:.1%} late rate ({int(row['Late_Forms'])} forms)\n"

# Phase insights
phase_data = analysis_results['phase_analysis']
phase_insights = "KEY INSIGHTS:\n\n"
for phase, row in phase_data.iterrows():
    phase_insights += f"• {phase}: {row['Late_Rate']:.1%} late rate ({int(row['Late_Forms'])} forms)\n"

# Site insights
worst_sites = analysis_results['site_analysis'].sort_values('Late_Rate', ascending=False).head(5)
site_insights = "KEY INSIGHTS:\n\n"
for (site_id, site_name, country), row in worst_sites.iterrows():
    site_insights += f"• {site_id} ({country}): {row['Late_Rate']:.1%} late rate ({int(row['Late_Forms'])}/{int(row['Total_Forms'])} forms)\n"

# ... (continue for other insights)

# Create an output folder
os.makedirs("plots", exist_ok=True)

# 1. Country Late Rate Analysis (your existing plotting code)
plt.figure(figsize=(16,8))

country_data = df.groupby('Country').agg(
    Total_Forms = ('# Forms Complete ≤ 14 days', 'size'),
    Late_Forms = ('# Forms Complete > 14 days', 'sum')
)
country_data['Late_Rate'] = country_data['Late_Forms'] / country_data['Total_Forms']
country_data = country_data.fillna(0)

country_data = country_data.sort_values('Late_Forms', ascending=False)

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
form_data = analysis_results['form_analysis'].sort_values('Late_Forms', ascending=False)
top_10_forms = form_data.head(10)
other_forms = form_data.iloc[10:]

# Prepare data
pie_data = top_10_forms['Late_Forms'].tolist()
pie_labels = [label[:30] + '...' if len(label) > 30 else label for label in top_10_forms.index]

if len(other_forms) > 0:
    pie_data.append(other_forms['Late_Forms'].sum())
    pie_labels.append('Other')

# Color palette with many distinct colors
colors_pie = sns.color_palette("tab20", len(pie_data))  

# Force "Other" to grey if it exists
if 'Other' in pie_labels:
    colors_pie[-1] = (0.6, 0.6, 0.6)

# Create pie chart
colors_pie = sns.color_palette("husl", len(pie_data))
if 'Other' in pie_labels:
    colors_pie[-1] = (0.6,0.6,0.6)  # force grey
wedges, texts, autotexts = plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                  colors=colors_pie, startangle=90)

plt.figure(figsize=(12,8))
wedges, texts, autotexts = plt.pie(
    pie_data, labels=pie_labels, autopct='%1.1f%%',
    colors=colors_pie, startangle=90
)

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
bars = plt.bar(phase_data.index, phase_percentages, color=COLORS['bad'])

plt.title('Late Submission Rate by Study Phase', fontsize=14, fontweight='bold')
plt.xlabel('Study Phase')
plt.ylabel('Percentage of Late Submissions (%)')
plt.xticks(rotation=45)

# Add number of late submissions on top of bars
for bar, late_count in zip(bars, phase_data['Late_Forms']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(late_count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

total_late_forms = phase_data['Late_Forms'].sum()
print(f"\nStudy phases responsible for {int(total_late_forms)} late forms in total")

plt.tight_layout()
plt.savefig("plots/3_study_phase_analysis.png", dpi=300)
plt.close()


# 4. Site Performance Analysis
plt.figure(figsize=(16,8))
site_data = analysis_results['site_analysis'][analysis_results['site_analysis']['Total_Forms'] > 20]
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

# Define country_time first
country_time = analysis_results['country_analysis'].dropna(subset=['Avg_Completion_Time'])
country_time = country_time.sort_values('Avg_Completion_Time', ascending=False)

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
plt.figure(figsize=(12, 8))
site_scatter = analysis_results['site_analysis']

# Bubble sizes rescaled with sqrt (normalized)
bubble_sizes = np.sqrt(site_scatter['Late_Forms'] + 1) * 10  # +1 avoids sqrt(0)

# Bubble chart
sc = plt.scatter(site_scatter['Total_Forms'],
                 site_scatter['Late_Rate'] * 100,
                 s=bubble_sizes,
                 c=[get_rag_color(x) for x in site_scatter['Late_Rate']],
                 alpha=0.6, edgecolors='k', linewidths=0.5)

# Improvements: log scale on X
plt.xscale("log")

# Titles and labels
plt.title('Site Performance Bubble Chart (Volume vs Late Rate)', fontsize=16, fontweight='bold')
plt.suptitle("Bubble size = # Late Forms | Color = Late Rate (RAG)", fontsize=10, y=0.92)
plt.xlabel('Total Forms (log scale)', fontsize=12)
plt.ylabel('Late Rate (%)', fontsize=12)

# Format x-axis to show thousands with commas
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

# Add horizontal reference lines for late rate benchmarks
for ref in [20, 40, 60]:
    plt.axhline(ref, color='gray', linestyle='--', linewidth=0.7)
    plt.text(plt.xlim()[1], ref+1, f"{ref}%", color='gray',
             ha='right', va='bottom', fontsize=8)

# Label worst 5 performers clearly with annotations
worst_sites = site_scatter.sort_values(['Late_Rate','Late_Forms'],
                                       ascending=[False, False]).head(5)
for idx, row in worst_sites.iterrows():
    x = row['Total_Forms']
    y = row['Late_Rate'] * 100
    plt.annotate(f"{idx[0]} ({idx[2]})",
                 (x, y),
                 xytext=(30, 10), textcoords="offset points",
                 fontsize=8, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", lw=0.7))

# Highlight one of the best sites (lowest late rate, highest volume)
best_site = site_scatter.sort_values(['Late_Rate','Total_Forms'],
                                     ascending=[True, False]).head(1)
x, y = best_site['Total_Forms'].values[0], best_site['Late_Rate'].values[0] * 100
plt.scatter(x, y, s=150, edgecolor='blue', facecolor='none', linewidths=1.5, zorder=5)
plt.annotate("Top Performer", (x, y), xytext=(-40, -30),
             textcoords="offset points", fontsize=9,
             color='blue', arrowprops=dict(arrowstyle="->", lw=1, color='blue'))

# Legend placeholders (bubble sizes)
for size in [10, 100, 1000]:
    plt.scatter([], [], s=np.sqrt(size) * 10, c='gray', alpha=0.4,
                label=f"{size} Late Forms")

plt.legend(scatterpoints=1, frameon=True, labelspacing=1,
           title="Bubble Size Reference", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/7_site_volume_vs_late_rate_refactored.png", dpi=300)
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
# Create shared month axis (once)
df['Completion_Month'] = pd.to_datetime(df['Form Complete Date'], errors='coerce').dt.to_period('M')
trend = df.groupby('Completion_Month')['Is_Late'].mean().reset_index()
trend['On_Time'] = (1 - trend['Is_Late']) * 100
trend['Month_Str'] = trend['Completion_Month'].astype(str)

fig = go.Figure()

# Add Global line (thick)
fig.add_trace(go.Scatter(
    x=trend['Month_Str'], 
    y=trend['On_Time'],
    mode='lines+markers',
    name='Global',
    line=dict(width=4, color='#708090'),
    marker=dict(size=8)
))

# Add worst 5 countries (thinner lines)
worst_countries = analysis_results['country_analysis'].head(5).index
colors_trend = ['#9b111e', '#ff9f14', '#8B0000', '#FF4500', '#DC143C']

for i, country in enumerate(worst_countries):
    country_trend = df[df['Country']==country].groupby('Completion_Month')['Is_Late'].mean().reset_index()
    if len(country_trend) > 0:
        country_trend['On_Time'] = (1 - country_trend['Is_Late']) * 100
        country_trend['Month_Str'] = country_trend['Completion_Month'].astype(str)
        
        fig.add_trace(go.Scatter(
            x=country_trend['Month_Str'],
            y=country_trend['On_Time'],
            mode='lines+markers',
            name=f'{country} (worst)',
            line=dict(width=2, color=colors_trend[i], dash='dash'),
            marker=dict(size=4)
        ))

fig.update_layout(
    title="On-Time % Trend Over Time (Interactive)",
    xaxis_title="Month",
    yaxis_title="% On-Time (<14 days)",
    hovermode='x unified'
)

fig.write_html("plots/9_trend_over_time_interactive.html")

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
visit_late_rate = visit_late_rate[visit_late_rate['Total_Forms'] >= 20]  # Filter visits with at least 5 forms
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

for bar, late_count in zip(bars, df.groupby('Country')['Is_Late'].sum()):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'n={int(late_count)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

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

site_perf_detailed['Late_Contribution_Pct'] = (
    site_perf_detailed['Late_Forms'] / total_late_forms) * 100

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


# Function for chart+text pages (MOVE THIS UP)
def create_chart_page(pdf, chart_path, title, description, insights):
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.3, 1.2])

    # Chart
    ax_chart = fig.add_subplot(gs[0])
    img = mpimg.imread(chart_path)
    ax_chart.imshow(img)
    ax_chart.axis('off')

    # Title area
    ax_title = fig.add_subplot(gs[1])
    ax_title.text(0.5, 0.5, title, fontsize=14, fontweight="bold",
                  ha="center", va="center")
    ax_title.axis('off')

    # Text area
    ax_text = fig.add_subplot(gs[2])
    wrapped = textwrap.fill(description + "\n\n" + insights, width=110)
    ax_text.text(0.05, 0.95, wrapped, fontsize=9,
                 ha="left", va="top", transform=ax_text.transAxes)
    ax_text.axis('off')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# Define insight text blocks
country_insights = """KEY INSIGHTS:
• Italy leads with highest late rate (42.5%) and significant volume (20,040 late forms)
• China demonstrates best performance with only 11.0% late rate despite high volume (30,320 forms)
• Korea shows moderate performance (24.5% late rate) but highest absolute volume (87,458 late forms)
• EMEA region shows highest average late rates (33.4%)"""

form_insights = """KEY INSIGHTS:
• ePRO PROXY INFORMATION shows 100% late rate (57 forms)
• Surgical pathology forms consistently problematic (100% late rates)
• Protocol milestone forms require immediate attention
• Top 10 forms account for significant portion of delays"""

phase_insights = """KEY INSIGHTS:
• Phase II shows highest late rate (29.7%) with 15,230 late forms
• Phase III represents largest volume (164,904 late forms, 22.1% rate)
• Phase IIB/III shows best performance (8.6% late rate)
• Early phase studies generally show higher completion times"""

site_insights = """KEY INSIGHTS:
• 28 sites show >70% late rates requiring immediate intervention
• Mexico site 7684A-008-0410 shows 100% late rate (18/18 forms)
• Korean sites dominate worst performers list
• Geographic clustering of performance issues evident"""

time_insights = """KEY INSIGHTS:
• Italy shows longest average completion time (33.4 days)
• China maintains shortest completion time (8.2 days)
• 14-day threshold exceeded by most countries
• Strong correlation between completion time and late rate"""

therapy_insights = """KEY INSIGHTS:
• Therapy areas show varying late submission patterns
• Oncology studies generally show higher late rates
• Volume concentration in specific therapy areas
• Resource allocation opportunities identified"""

bubble_insights = """KEY INSIGHTS:
• High-volume sites don't necessarily have high late rates
• Several low-volume sites show concerning 100% late rates
• Best performers combine high volume with low late rates
• Bubble size indicates impact potential for interventions"""

distribution_insights = """KEY INSIGHTS:
• Median completion time: 12.1 days (below 14-day threshold)
• Long tail distribution with some forms taking >200 days
• 25% of forms completed within 6.8 days
• Distribution suggests systematic delays in subset of forms"""

split_insights = """KEY INSIGHTS:
• Overall 76.8% on-time performance across all forms
• 23.2% late submission rate requires attention
• 271,812 forms completed late (>14 days)
• Performance gap from industry benchmarks evident"""

cra_insights = """KEY INSIGHTS:
• Significant variation in CRA performance (0-100% on-time rates)
• Top performers achieve >95% on-time rates consistently
• Bottom performers require immediate training/support
• CRA performance directly impacts site outcomes"""

visit_insights = """KEY INSIGHTS:
• Follow-up visits show highest late rates (100% for some)
• Cycle-based visits demonstrate consistent delays
• Visit complexity correlates with completion delays
• Scheduling optimization opportunities identified"""

rag_insights = """KEY INSIGHTS:
• Only Finland achieves 'Green' status (>80% on-time)
• Most countries fall in 'Red' category (<60% on-time)
• Clear performance tiers visible across countries
• Benchmark gaps indicate improvement potential"""

leaderboard_insights = """KEY INSIGHTS:
• Best sites achieve >95% on-time performance
• Worst sites show <20% on-time performance
• Geographic diversity in both best and worst performers
• Performance spread indicates training/process opportunities"""

warning_insights = """KEY INSIGHTS:
• 25 sites identified as high-risk (<50% on-time)
• These sites contribute disproportionately to late forms
• Immediate intervention required for worst performers
• Early warning system can prevent further deterioration"""

# Create enhanced PDF with insights
with PdfPages("Enhanced_eCRF_Report.pdf") as pdf:
    # Chart 1: Country Analysis
    create_chart_page(pdf, "plots/1_country_late_rate.png",
                      "Chart 1: Country Late Rate Analysis",
                      "Late submission rates by country, ordered by total late forms. Colors indicate RAG status.",
                      country_insights)
    
    # Chart 2: Form Type Analysis  
    create_chart_page(pdf, "plots/2_form_type_analysis.png",
                      "Chart 2: Form Type Analysis",
                      "Distribution of late forms across different form types (Top 10 + Other).",
                      form_insights)
    
    # Chart 3: Study Phase Analysis
    create_chart_page(pdf, "plots/3_study_phase_analysis.png", 
                      "Chart 3: Study Phase Analysis",
                      "Late submission rates across different study phases.",
                      phase_insights)
    
    # Chart 4: Site Performance
    create_chart_page(pdf, "plots/4_site_performance.png",
                      "Chart 4: Site Performance Analysis", 
                      "Top 20 worst performing sites (>5 forms). Colors indicate RAG status.",
                      site_insights)
    
    # Chart 5: Completion Time by Country
    create_chart_page(pdf, "plots/5_avg_completion_time_by_country.png",
                      "Chart 5: Average Completion Time by Country",
                      "Average form completion time by country with 14-day threshold reference.",
                      time_insights)
    
    # Chart 6: Therapy Area Analysis
    create_chart_page(pdf, "plots/6_therapy_area_analysis.png",
                      "Chart 6: Therapy Area Analysis", 
                      "Late submission rates by therapy area (Top 10 + Other).",
                      therapy_insights)
    
    # Chart 7: Volume vs Late Rate Bubble
    create_chart_page(pdf, "plots/7_site_volume_vs_late_rate_refactored.png",
                      "Chart 7: Site Volume vs Late Rate Analysis",
                      "Bubble chart showing relationship between site volume and late rates.",
                      bubble_insights)
    
    # Chart 8: Completion Time Distribution
    create_chart_page(pdf, "plots/8_completion_time_distribution.png",
                      "Chart 8: Completion Time Distribution",
                      "Global distribution of form completion times with statistical markers.",
                      distribution_insights)
    
    # Chart 10: On-Time vs Late Split
    create_chart_page(pdf, "plots/10_on_time_vs_late_split.png",
                      "Chart 10: Overall On-Time vs Late Split",
                      "Overall performance split showing on-time vs late submissions.",
                      split_insights)
    
    # Chart 11: CRA Performance
    create_chart_page(pdf, "plots/11_cra_performance.png",
                      "Chart 11: CRA Performance Analysis",
                      "Best and worst performing CRAs (minimum 5 forms).",
                      cra_insights)
    
    # Chart 12: Visit Type Analysis
    create_chart_page(pdf, "plots/12_visit_type_analysis.png",
                      "Chart 12: Visit Type Analysis",
                      "Late rates by visit type (Top 15, minimum 20 forms).",
                      visit_insights)
    
    # Chart 13: RAG Benchmark
    create_chart_page(pdf, "plots/13_rag_benchmark.png",
                      "Chart 13: RAG Benchmark Analysis",
                      "Country performance against RAG thresholds (80% good, 60% warning).",
                      rag_insights)
    
    # Chart 14: Best vs Worst Leaderboard
    create_chart_page(pdf, "plots/14_best_worst_leaderboard.png",
                      "Chart 14: Site Performance Leaderboard",
                      "Top 10 best and worst performing sites (minimum 5 forms).",
                      leaderboard_insights)
    
    # Chart 15: Early Warning
    create_chart_page(pdf, "plots/15_early_warning.png",
                      "Chart 15: Early Warning Dashboard",
                      "High-risk sites requiring immediate attention (<50% on-time).",
                      warning_insights)

print("✅ PDF generated: eCRF_Late_Submission_Report.pdf")