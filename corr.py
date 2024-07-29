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
uploaded_file = st.file_uploader("Upload a CSV file", type=("xlsx","csv", "xls"))
if uploaded_file:
    try:
        # Try to read the data with different encodings and headers
        data = pd.read_csv(uploaded_file, encoding='utf-8', header=None)
        if data.empty:
            raise ValueError("The file is empty or could not be parsed.")
        
        # Display the first few rows to understand the structure
        st.write("Data preview:")
        st.write(data.head())

        # Check if the first row should be used as headers
        if not data.columns.str.contains('Unnamed').all():
            data.columns = data.iloc[0]  # Use the first row as column headers
            data = data[1:]  # Remove the header row from the data
        else:
            # Reset the column names if no meaningful header was found
            data.columns = [f'Column_{i}' for i in range(data.shape[1])]

        # Check for empty data
        if data.empty:
            st.error("The data does not contain any columns after header adjustment.")
            st.stop()

        # Convert columns to numeric, if possible
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

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

    except ValueError as ve:
        st.error(f"An error occurred: {ve}")
    except pd.errors.EmptyDataError:
        st.error("No columns to parse from file. The file might be empty or improperly formatted.")
    except UnicodeDecodeError:
        st.error("Unable to decode the file. Please check the file encoding.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
