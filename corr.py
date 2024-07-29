import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Function to calculate significance of correlation
def correlation_significance(data):
    columns = data.columns
    num_vars = len(columns)
    p_values = pd.DataFrame(np.ones((num_vars, num_vars)), columns=columns, index=columns)
    
    for i in range(num_vars):
        for j in range(num_vars):
            if i != j:
                _, p_value = pearsonr(data[columns[i]], data[columns[j]])
                p_values.iloc[i, j] = p_value
    
    return p_values

# Streamlit app
st.title('Correlation Analysis Tool')

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    # Read the data
    data = pd.read_csv(uploaded_file)
    
    st.write("Data preview:")
    st.write(data.head())

    # Compute correlation matrix
    corr_matrix = data.corr(method='pearson')
    st.write("Correlation Matrix:")
    st.write(corr_matrix)

    # Compute significance levels
    p_values = correlation_significance(data)
    st.write("P-values for Correlation Significance:")
    st.write(p_values)

    # Plot Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot(plt.gcf())
    plt.clf()

    # Plot pairwise scatter plots
    st.subheader("Pairwise Scatter Plots")
    pair_plot = sns.pairplot(data)
    st.pyplot(pair_plot.figure)
    plt.clf()

    # Plot correlation matrix with significance
    st.subheader("Correlation Heatmap with Significance")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=p_values > 0.05)
    plt.title('Correlation Heatmap with Significance Mask')
    st.pyplot(plt.gcf())
    plt.clf()
