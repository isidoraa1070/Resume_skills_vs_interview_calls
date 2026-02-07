# utils/feature_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(csv_path: str):
    """
    Load dataset and map binary columns to numeric.

    """
    data = pd.read_csv(csv_path)
    
    # Map binary columns to 0/1
    binary_map = {"Yes": 1, "No": 0}
    for col in ["internship", "github_portfolio"]:
        if col in data.columns:
            data[col] = data[col].map(binary_map)
    
    return data

def analyze_degrees(data: pd.DataFrame):
    """
    Analyze descriptive stats for each degree.

    """
    
    # Descriptive statistics per degree
    desc_stats = data.groupby("degree")["interview_calls"].describe()


    desc_stats.to_excel("excel/descriptive_stats_by_degree.xlsx")

def analyze_numeric_features(data: pd.DataFrame):
    """
    Compute correlations between numeric features and interview_calls,
    and save Pearson correlation matrix.

    """
    numeric_data = data.select_dtypes(include=["number", "bool"])
    
    # Correlation with interview_calls
    corr_with_target = numeric_data.corr()["interview_calls"].sort_values(ascending=False)
    corr_with_target.to_frame(name="corr_with_interview_calls").to_excel("excel/correlation_with_interview_calls.xlsx")
    
    # Pearson correlation matrix
    pearson_matrix = numeric_data.corr()
    sns.heatmap(pearson_matrix, annot=True, cmap="coolwarm")
    plt.show()
    

def plot_degree_boxplot(data: pd.DataFrame):
    """
    Plot a boxplot of interview calls by degree.

    """
    plt.figure(figsize=(8,6))
    sns.boxplot(x="degree", y="interview_calls", data=data)
    plt.title("Interview Calls by Degree")
    plt.xlabel("Degree")
    plt.ylabel("Interview Calls")
    

    plt.show()


if __name__ == "__main__":
    data = load_and_prepare_data("data/resume_skills_vs_interview_calls.csv")
    

    analyze_degrees(data)
    
    
    analyze_numeric_features(data)
    
   
    plot_degree_boxplot(data)



