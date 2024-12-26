import pandas as pd
import numpy as np
#Library for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from plotly.offline import iplot
from plotly.subplots import make_subplots
import missingno as msno
from pandas.plotting import parallel_coordinates
import altair as alt
#Library for building machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
# Full path to the CSV file
file_path = r'C:\Users\mario\Desktop\hr_dashboard_data.csv' # Use raw string by prefixing with r
# Read the CSV file
df = pd.read_csv(file_path)
# Display the first few rows of the dataframe
print(df.head())
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])
duplicated_data = df.duplicated().any()
print(duplicated_data)
df.describe()
print(df.describe())
df.isnull().sum()
print(df.isnull().sum())
df.dtypes
print(df.dtypes)
df.hist(figsize=(18,10))
plt.show()
#Age Distribution 
histogram_trace = go.Histogram(
 x=df['Age'],
 nbinsx=10, # Number of bins
 histnorm='percent', # Normalize histogram to percentage
 name='Age Distribution'
)
# Create the figure and add the trace
fig = make_subplots(rows=1, cols=1)
fig.add_trace(histogram_trace)
# Update layout of the plot
fig.update_layout(
 title='Age Distribution of Employees',
 xaxis_title='Age',
 yaxis_title='Percent',
 template='plotly',
 width=800,
 height=500
)
# Display the figure
fig.show()
#Gender Distribution
# Create a copy of the DataFrame and count gender distribution
gender_plot = df['Gender'].value_counts()
# Create a subplot for the pie chart
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'pie'}]], subplot_titles=['Gender Distribution'])
# Add the pie chart trace
fig.add_trace(go.Pie(values=gender_plot.values, labels=gender_plot.index), row=1, col=1)
# Update the chart layout and design
fig.update_traces(hoverinfo='label', textfont_size=18, textposition='auto',
 marker=dict(colors=["#9467BD", "#1F77B4"], line=dict(color='white', width=2)))
fig.update_layout(title="<b>Gender Distribution of the Employees</b>", title_x=0.5, title_y=0.95,
 template='xgridoff', width=800, height=500)
# Display the figure
fig.show()
#Employee Distribution by Department and Position
fig = px.treemap(
 df,
 path=['Department','Position'],
 values=[1] * len(df),
 title='Employee Distribution by Department and Position'
)
fig.show()
#Average Productivity and Satisfaction rate by Department and Position
avg_metric_dept = df.groupby('Department')[['Productivity (%)', 'Satisfaction Rate (%)']].mean().reset_index()
avg_metric_pos = df.groupby('Position')[['Productivity (%)', 'Satisfaction Rate (%)']].mean().reset_index()
fig_dept = px.bar(
 avg_metric_dept,
 x='Department',
 y=['Productivity (%)', 'Satisfaction Rate (%)'],
 title='Average Metrics by Department',
 labels={'Department': 'Department'},
 height=500
)
fig_dept.update_layout(barmode='group')
fig_pos = px.bar(
 avg_metric_pos,
 x='Position',
 y=['Productivity (%)', 'Satisfaction Rate (%)'],
 title='Average Metrics by Position',
 labels={'Position': 'Position'},
 height=500
)
fig_pos.update_layout(barmode='group')
fig_dept.show()
fig_pos.show()
#Salary vs. Performance/Satisfaction
# Scatter plot for Salary vs Productivity
fig_prod = px.scatter(
 df,
 x='Salary',
 y='Productivity (%)',
 title='Salary vs Productivity',
 trendline='ols', # Adds a trendline using Ordinary Least Squares (OLS)
 labels={'Salary': 'Salary', 'Productivity (%)': 'Productivity (%)'},
 height=500
)
# Scatter plot for Salary vs Satisfaction Rate
fig_satis = px.scatter(
 df,
 x='Salary',
 y='Satisfaction Rate (%)',
 title='Salary vs Satisfaction Rate',
 trendline='ols', # Adds a trendline using Ordinary Least Squares (OLS)
 labels={'Salary': 'Salary', 'Satisfaction Rate (%)': 'Satisfaction Rate (%)'},
 height=500
)
# Display the plots
fig_prod.show()
fig_satis.show()
salary_ranges = pd.cut(df['Salary'], bins=[0, 40000, 60000, 80000, 100000, float('inf')],
 labels=['<40k', '40k-60k', '60k-80k', '80k-100k', '100k+'])
average_metrics_by_salary = df.groupby(salary_ranges)[['Productivity (%)', 'Satisfaction Rate (%)']].mean().reset_index()
print(average_metrics_by_salary)
fig_avg_metric = px.bar(
 average_metrics_by_salary,
 x='Salary',
 y=['Productivity (%)', 'Satisfaction Rate (%)'],
 title='Average Metrics by Salary Range',
 labels={'Salary': 'Salary Range', 'value': 'Average Rate (%)'},
 height=500
)
fig_avg_metric.show()
# Feedback Scores Distribution 
fig = px.histogram(df, x='Feedback Score', nbins=5, title='Distribution of Feedback Scores')
fig.update_layout(xaxis_title='Feedback Score', yaxis_title='Frequency')
fig.show()
# Correlation between Feedback Scores and Other Metrics
sns.set(style="whitegrid") # Set the style to whitegrid
g = sns.pairplot(df, x_vars=['Feedback Score'], y_vars=['Productivity (%)', 'Satisfaction Rate (%)'], 
kind='scatter')
# Adjust the layout of the pairplot
g.fig.suptitle('Correlation between Feedback Scores and Metrics', y=1.02) # Set the title above the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust the layout to avoid title overlap
plt.show()
