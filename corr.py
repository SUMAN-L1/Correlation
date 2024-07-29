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

# Function to attempt reading a CSV file with different encodings
def read_csv_with_encodings(file, encodings=['utf-8', 'latin1', 'ISO-8859-1']):
    for encoding in encodings:
        try:
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the file with the provided encodings.")

# Function to read data from uploaded file
def read_file(file, file_type):
    if file_type == 'csv':
        return read_csv_with_encodings(file)
    elif file_type == 'xlsx':
        try:
            return pd.read_excel(file)
        except Exception as e:
            raise ValueError(f"Unable to read the Excel file. Error: {e}")
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

# Streamlit app
st.title('Correlation Analysis Tool')

# Upload file
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    file_type = uploaded_file.type.split('/')[1]

    try:
        # Read the file
        data = read_file(uploaded_file, file_type)
        if data.empty:
            raise ValueError("The file is empty or could not be parsed.")
        
        # Display the first few rows to understand the structure
        st.write("Data preview:")
        st.write(data.head())

        # Check if the first row should be used as headers
        if data.columns[0] == 'Unnamed: 0':
            data.columns = data.iloc[0]  # Use the first row as column headers
            data = data[1:]  # Remove the header row from the data

        # Reset the index
        data.reset_index(drop=True, inplace=True)

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
        st.error("Unable to decode the file. Please check the file encoding or use a different encoding.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
