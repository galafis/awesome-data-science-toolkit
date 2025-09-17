#!/usr/bin/env python3
"""
Awesome Data Science Toolkit - Quick Start Example

This script demonstrates the basic functionality of our data science toolkit.
It showcases data loading, preprocessing, visualization, and machine learning.

Author: Gabriel Demetrios Lafis
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def main():
    """
    Main function demonstrating the data science toolkit capabilities.
    """
    print("🚀 Welcome to the Awesome Data Science Toolkit!")
    print("=" * 50)
    
    # 1. Data Loading and Exploration
    print("\n📊 Step 1: Loading and exploring data...")
    
    # Load the classic Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(df.describe())
    
    # 2. Data Visualization
    print("\n📈 Step 2: Creating visualizations...")
    
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Correlation heatmap
    correlation = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
    axes[0, 0].set_title('Feature Correlation Matrix')
    
    # Species distribution
    df['species'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['skyblue', 'lightgreen', 'salmon'])
    axes[0, 1].set_title('Species Distribution')
    axes[0, 1].set_xlabel('Species')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Pairplot substitute - scatter plot
    scatter = axes[1, 0].scatter(df['sepal length (cm)'], df['sepal width (cm)'], 
                                c=df['target'], cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Sepal Length (cm)')
    axes[1, 0].set_ylabel('Sepal Width (cm)')
    axes[1, 0].set_title('Sepal Length vs Width by Species')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Box plot
    df.boxplot(column='petal length (cm)', by='species', ax=axes[1, 1])
    axes[1, 1].set_title('Petal Length Distribution by Species')
    axes[1, 1].set_xlabel('Species')
    axes[1, 1].set_ylabel('Petal Length (cm)')
    
    plt.tight_layout()
    plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'iris_analysis.png'")
    
    # 3. Machine Learning Pipeline
    print("\n🤖 Step 3: Training machine learning models...")
    
    # Prepare features and target
    X = df.drop(['target', 'species'], axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Feature Importance:")
    print(feature_importance)
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # 4. Advanced Example with Synthetic Data
    print("\n🔬 Step 4: Advanced example with synthetic data...")
    
    # Generate synthetic dataset
    X_syn, y_syn = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    
    # Split and scale
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
        X_syn, y_syn, test_size=0.2, random_state=42
    )
    
    scaler_syn = StandardScaler()
    X_syn_train_scaled = scaler_syn.fit_transform(X_syn_train)
    X_syn_test_scaled = scaler_syn.transform(X_syn_test)
    
    # Train model
    rf_syn = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_syn.fit(X_syn_train_scaled, y_syn_train)
    
    # Evaluate
    y_syn_pred = rf_syn.predict(X_syn_test_scaled)
    syn_accuracy = accuracy_score(y_syn_test, y_syn_pred)
    
    print(f"🎯 Synthetic Data Model Accuracy: {syn_accuracy:.4f} ({syn_accuracy*100:.2f}%)")
    
    print("\n✨ Quick start completed successfully!")
    print("🔗 Check out our other examples and tutorials for more advanced features.")
    print("📚 Visit the documentation for detailed guides and API reference.")
    
    # Show the plot if running interactively
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
