import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
 
 
df = pd.read_csv(r"C:\Users\a239584\Downloads\Coding\eCRF_data\SEE Report 03Sep2025_CLEANED.csv")
 
 
df['Form Complete Date'] = pd.to_datetime(df['Form Complete Date'], errors='coerce')
 
 
df = df[df['Form Complete Date'].notna()]
six_months_ago = pd.Timestamp.now() - pd.Timedelta(days=180)
df = df[df['Form Complete Date'] >= six_months_ago]
 
 
df['Is_Late'] = df['# Forms Complete > 14 days'] > 0
df['Late_Rate'] = df['# Forms Complete > 14 days'] / (
    df['# Forms Complete ≤ 14 days'] + df['# Forms Complete > 14 days']
)
df['AVG_Form_Complete_CT_Clean'] = pd.to_numeric(df['AVG Form Complete CT'], errors='coerce')
 
 
country_dropdown = widgets.Dropdown(
    options=['All'] + sorted(df['Country'].dropna().unique().tolist()),
    description='Country:',
    value='All'
)
 
phase_dropdown = widgets.Dropdown(
    options=['All'] + sorted(df['Study Phase'].dropna().unique().tolist()),
    description='Phase:',
    value='All'
)
 
cra_dropdown = widgets.Dropdown(
    options=['All'] + sorted(df['CRA'].dropna().unique().tolist()),
    description='CRA:',
    value='All'
)
 
form_dropdown = widgets.Dropdown(
    options=['All'] + sorted(df['Form Name'].dropna().unique().tolist()),
    description='Form:',
    value='All'
)
 
 
def filter_data():
    filtered_df = df.copy()
    if country_dropdown.value != 'All':
        filtered_df = filtered_df[filtered_df['Country'] == country_dropdown.value]
    if phase_dropdown.value != 'All':
        filtered_df = filtered_df[filtered_df['Study Phase'] == phase_dropdown.value]
    if cra_dropdown.value != 'All':
        filtered_df = filtered_df[filtered_df['CRA'] == cra_dropdown.value]
    if form_dropdown.value != 'All':
        filtered_df = filtered_df[filtered_df['Form Name'] == form_dropdown.value]
    return filtered_df
 
def update_dashboard(change=None):
    filtered_df = filter_data()
 
 
    country_perf = filtered_df.groupby('Country').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).reset_index()
    country_perf.columns = ['Country', 'Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    fig1 = px.bar(country_perf.sort_values('Late_Rate', ascending=False).head(10),
                  x='Country', y='Late_Rate', title='Top 10 Countries by Late Rate')
    fig1.show()
 
 
    form_perf = filtered_df.groupby('Form Name').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).reset_index()
    form_perf.columns = ['Form Name', 'Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    fig2 = px.bar(form_perf.sort_values('Late_Rate', ascending=False).head(10),
                  x='Form Name', y='Late_Rate', title='Top 10 Form Types by Late Rate')
    fig2.show()
 
    phase_perf = filtered_df.groupby('Study Phase').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).reset_index()
    phase_perf.columns = ['Study Phase', 'Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    fig3 = px.bar(phase_perf.sort_values('Late_Rate', ascending=False),
                  x='Study Phase', y='Late_Rate', title='Late Rate by Study Phase')
    fig3.show()
 
 
    visit_perf = filtered_df.groupby('Visit Name').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).reset_index()
    visit_perf.columns = ['Visit Name', 'Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    fig4 = px.bar(visit_perf.sort_values('Late_Rate', ascending=False).head(10),
                  x='Visit Name', y='Late_Rate', title='Top 10 Visit Types by Late Rate')
    fig4.show()
 
 
    site_perf = filtered_df.groupby('Site Name (CTMS)').agg({
        'Is_Late': ['count', 'sum', 'mean'],
        'AVG_Form_Complete_CT_Clean': 'mean'
    }).reset_index()
    site_perf.columns = ['Site Name (CTMS)', 'Total_Forms', 'Late_Forms', 'Late_Rate', 'Avg_Completion_Time']
    fig5 = px.bar(site_perf.sort_values('Late_Rate', ascending=False).head(10),
                  x='Site Name (CTMS)', y='Late_Rate', title='Top 10 Sites by Late Rate')
    fig5.show()
 
 
    fig6 = px.histogram(filtered_df, x='AVG_Form_Complete_CT_Clean', nbins=50,
                        title='Completion Time Distribution')
    fig6.show()
 
 
country_dropdown.observe(update_dashboard, names='value')
phase_dropdown.observe(update_dashboard, names='value')
cra_dropdown.observe(update_dashboard, names='value')
form_dropdown.observe(update_dashboard, names='value')
 
 
display(country_dropdown, phase_dropdown, cra_dropdown, form_dropdown)
update_dashboard()