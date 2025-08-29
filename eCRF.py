import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv(r"C:\Users\a239584\Downloads\Coding\eCRF_data\Data Entry compliance 14May2025 all dimentions.csv")

# Display basic info about the dataset
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

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

"""
# Create comprehensive visualizations
plt.style.use('default')
fig = plt.figure(figsize=(20, 24))
"""

# Create an output folder
os.makedirs("plots", exist_ok=True)

# 1. Country Late Rate Analysis
plt.figure(figsize=(16,8))
country_data = analysis_results['country_analysis']  # Show all countries instead of just top 10
bars = plt.bar(country_data.index, country_data['Late_Rate'],
               color=['red' if x > 0.5 else 'orange' if x > 0.3 else 'green'
                      for x in country_data['Late_Rate']])
plt.title('Late Submission Rate by Country (All Countries)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Late Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/1_country_late_rate.png", dpi=300)
plt.close()


# 2. Form Type Analysis (Top 10 most problematic)
plt.figure(figsize=(12,8))
form_data = analysis_results['form_analysis'].head(10)
plt.barh(form_data.index, form_data['Late_Rate'],
         color=['red' if x == 1.0 else 'orange' if x > 0.5 else 'yellow' 
                for x in form_data['Late_Rate']])
plt.title('Top 10 Most Problematic Form Types', fontsize=14, fontweight='bold')
plt.xlabel("Late Rate")
plt.tight_layout()
plt.savefig("plots/2_form_type_analysis.png", dpi=300)
plt.close()


# 3. Study Phase Analysis
plt.figure(figsize=(10,6))
phase_data = analysis_results['phase_analysis']
bars = plt.bar(phase_data.index, phase_data['Late_Rate'],
               color=['red' if x > 0.7 else 'orange' if x > 0.4 else 'green'
                      for x in phase_data['Late_Rate']])
plt.title('Late Submission Rate by Study Phase', fontsize=14, fontweight='bold')
plt.ylabel('Late Rate')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig("plots/3_study_phase_analysis.png", dpi=300)
plt.close()


# 4. Site Performance Analysis
plt.figure(figsize=(16,8))
site_data = analysis_results['site_analysis'][analysis_results['site_analysis']['Total_Forms'] > 5]
site_data = site_data.sort_values('Late_Rate', ascending=False).head(20)  # show only top 20
bars = plt.bar(range(len(site_data)), site_data['Late_Rate'],
               color=['red' if x > 0.7 else 'orange' if x > 0.4 else 'green'
                      for x in site_data['Late_Rate']])
plt.title('Site Performance (Top 20, >5 forms)', fontsize=14, fontweight='bold')
plt.ylabel('Late Rate')
# Updated labels to include country
plt.xticks(range(len(site_data)),
           [f"{idx[0]}-{idx[2]}" for idx in site_data.index],  # Site-Country format
           rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig("plots/4_site_performance.png", dpi=300)
plt.close()


# 5. Average Completion Time by Country
plt.figure(figsize=(16,8))
country_time = analysis_results['country_analysis'].dropna(subset=['Avg_Completion_Time'])  # Show all countries
bars = plt.bar(country_time.index, country_time['Avg_Completion_Time'],
               color=['red' if x > 200 else 'orange' if x > 100 else 'green'
                      for x in country_time['Avg_Completion_Time']])
plt.title('Average Form Completion Time by Country (All Countries)', fontsize=14, fontweight='bold')
plt.ylabel('Avg Completion Time (days)')
plt.xticks(rotation=45, ha='right')
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
bars = plt.bar(therapy_analysis.index, therapy_analysis['Late_Rate'] * 100,  # Convert to percentage
               color=['red' if x > 0.5 else 'orange' if x > 0.3 else 'green'
                      for x in therapy_analysis['Late_Rate']])
plt.title('Percentage of Late Submissions by Therapy Area', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Late Submissions (%)')  # Updated label
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("plots/6_therapy_area_analysis.png", dpi=300)
plt.close()


# 7. Volume vs Late Rate Scatter (Sites)
plt.figure(figsize=(12,8))
site_scatter = analysis_results['site_analysis']
# Create size categories for better visualization
sizes = []
for forms in site_scatter['Total_Forms']:
    if forms < 10:
        sizes.append(30)
    elif forms < 50:
        sizes.append(60)
    elif forms < 100:
        sizes.append(90)
    else:
        sizes.append(120)

scatter = plt.scatter(site_scatter['Total_Forms'], site_scatter['Late_Rate'],
                     s=sizes, alpha=0.6, c=site_scatter['Late_Rate'], cmap='RdYlGn_r')
plt.title('Site Volume vs Late Rate\n(Bubble size = Volume category)', fontsize=14, fontweight='bold')
plt.xlabel('Total Forms')
plt.ylabel('Late Rate')
plt.colorbar(scatter, label='Late Rate')

# Add quadrant lines for easier interpretation
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='50% Late Rate')
plt.axvline(x=site_scatter['Total_Forms'].median(), color='gray', linestyle=':', alpha=0.7, 
           label=f'Median Volume ({site_scatter["Total_Forms"].median():.0f})')

# Trend line
z = np.polyfit(site_scatter['Total_Forms'], site_scatter['Late_Rate'], 1)
p = np.poly1d(z)
plt.plot(site_scatter['Total_Forms'], p(site_scatter['Total_Forms']),
         "r--", alpha=0.8, label='Trend line')

plt.legend()
plt.tight_layout()
plt.savefig("plots/7_site_volume_vs_late_rate.png", dpi=300)
plt.close()


# 8. Completion Time Distribution
plt.figure(figsize=(10,6))
completion_times = df['AVG_Form_Complete_CT_Clean'].dropna()
# Filter to 0-200 days range
completion_times_filtered = completion_times[(completion_times >= 0) & (completion_times <= 200)]
plt.hist(completion_times_filtered, bins=40, alpha=0.7, color='skyblue', edgecolor='black', range=(0, 200))
plt.axvline(x=14, color='red', linestyle='--', linewidth=2, label='14-day threshold')
plt.title('Distribution of Form Completion Times (0-200 days)', fontsize=14, fontweight='bold')
plt.xlabel('Completion Time (days)')
plt.ylabel('Frequency')
plt.xlim(0, 200)  # Set x-axis range
plt.legend()
plt.tight_layout()
plt.savefig("plots/8_completion_time_distribution.png", dpi=300)
plt.close()

# 9. Trend Over Time (Global, Country, Site)
plt.figure(figsize=(14,8))
df['Completion_Month'] = pd.to_datetime(df['Form Complete Date'], errors='coerce').dt.to_period('M')
trend = df.groupby('Completion_Month')['Is_Late'].mean().reset_index()
trend['On_Time'] = 1 - trend['Is_Late']

# Plot global trend
plt.plot(range(len(trend)), trend['On_Time']*100, marker='o', linewidth=2, label="Global", markersize=6)

# Add 5 worst performing countries
worst_countries = analysis_results['country_analysis'].head(5).index
colors = ['red', 'orange', 'purple', 'brown', 'pink']
for i, country in enumerate(worst_countries):
    trend_country = df[df['Country']==country].groupby('Completion_Month')['Is_Late'].mean().reset_index()
    if len(trend_country) > 0:
        trend_country['On_Time'] = 1 - trend_country['Is_Late']
        plt.plot(range(len(trend_country)), trend_country['On_Time']*100, 
                marker='s', label=f'{country} (worst)', color=colors[i], markersize=4)

plt.title("On-Time % Trend Over Time (Monthly Averages)", fontsize=14, fontweight="bold")
plt.ylabel("% On-Time (<14 days)")
plt.xlabel("Month Index")
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

bars = plt.bar(categories, values, color=['green','red','gray'])
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
plt.figure(figsize=(14,6))
cra_perf = df.groupby('CRA')['Is_Late'].mean().reset_index()
cra_perf['On_Time%'] = (1 - cra_perf['Is_Late'])*100
cra_perf = cra_perf.sort_values('On_Time%', ascending=True).tail(20)  # last 20

sns.barplot(x='On_Time%', y='CRA', data=cra_perf, palette="RdYlGn")
plt.title("CRA On-Time Performance (%)", fontsize=14, fontweight="bold")
plt.xlabel("% On-Time (<14 days)")
plt.ylabel("CRA")
plt.tight_layout()
plt.savefig("plots/11_cra_performance.png", dpi=300)
plt.close()

# 12. Visit Type Analysis
plt.figure(figsize=(14,6))
visit_perf = df.groupby('Visit Name')['AVG_Form_Complete_CT_Clean'].mean().sort_values().head(15)
sns.barplot(x=visit_perf.values, y=visit_perf.index, palette="Blues_r")
plt.title("Average Completion Time by Visit Type", fontsize=14, fontweight="bold")
plt.xlabel("Avg Completion Time (days)")
plt.ylabel("Visit Type")
plt.tight_layout()
plt.savefig("plots/12_visit_type_analysis.png", dpi=300)
plt.close()

# 13. RAG Benchmark (Countries)
plt.figure(figsize=(16,8))
country_perf = df.groupby('Country')['Is_Late'].mean().reset_index()  # All countries
country_perf['On_Time%'] = (1 - country_perf['Is_Late'])*100

colors = country_perf['On_Time%'].apply(
    lambda x: 'green' if x >= 80 else ('orange' if x >= 60 else 'red')
)

plt.bar(country_perf['Country'], country_perf['On_Time%'], color=colors)
plt.title("RAG Benchmark: On-Time Submissions by Country (All Countries)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha='right')
plt.ylabel("% On-Time (<14 days)")
plt.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% threshold')
plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='60% threshold')
plt.legend()
plt.tight_layout()
plt.savefig("plots/13_rag_benchmark.png", dpi=300)
plt.close()

# 14. Best vs Worst Sites Leaderboard
plt.figure(figsize=(14,8))
site_perf = df.groupby('Study Site')['Is_Late'].mean().reset_index()
site_perf['On_Time%'] = (1 - site_perf['Is_Late'])*100
best = site_perf.sort_values('On_Time%', ascending=False).head(10)
worst = site_perf.sort_values('On_Time%').head(10)
leaderboard = pd.concat([best, worst])

sns.barplot(x='On_Time%', y='Study Site', data=leaderboard, palette="RdYlGn")
plt.title("Leaderboard: Best & Worst Performing Sites", fontsize=14, fontweight="bold")
plt.xlabel("% On-Time")
plt.tight_layout()
plt.savefig("plots/14_best_worst_leaderboard.png", dpi=300)
plt.close()

# 15. Early Warning Dashboard (simple rule)
plt.figure(figsize=(14,8))
site_perf_detailed = df.groupby(['Study Site', 'Country'])['Is_Late'].mean().reset_index()
site_perf_detailed['On_Time%'] = (1 - site_perf_detailed['Is_Late'])*100
site_perf_detailed['Site_Country'] = site_perf_detailed['Study Site'] + ' (' + site_perf_detailed['Country'] + ')'

# Filter high risk sites and sort by risk level (lowest On_Time% first)
risk_sites = site_perf_detailed[site_perf_detailed['On_Time%'] < 50].sort_values('On_Time%')

if len(risk_sites) > 0:
    sns.barplot(x='On_Time%', y='Site_Country', data=risk_sites, color="red")
    plt.title("Early Warning: High Risk Sites (<50% On-Time) - Ordered by Risk Level", fontsize=14, fontweight="bold")
    plt.xlabel("% On-Time")
else:
    plt.text(0.5, 0.5, 'No sites with <50% On-Time performance', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    plt.title("Early Warning: High Risk Sites (<50% On-Time)", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("plots/15_early_warning.png", dpi=300)
plt.close()

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