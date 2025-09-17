"""
Awesome Data Science Toolkit - Machine Learning Module

This module provides ready-to-use machine learning algorithms with clean,
documented code. It includes implementations for:
- Classification algorithms
- Regression algorithms  
- Clustering algorithms
- Deep learning models

Author: Gabriel Demetrios Lafis
Date: September 2025
"""

from .classification import (
    LogisticRegressionModel,
    RandomForestClassifierModel,
    SVMModel,
    NeuralNetworkClassifier
)

from .regression import (
    LinearRegressionModel,
    PolynomialRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel
)

from .clustering import (
    KMeansModel,
    DBSCANModel,
    HierarchicalClusteringModel
)

from .deep_learning import (
    CNNModel,
    RNNModel,
    LSTMModel
)

__all__ = [
    # Classification
    'LogisticRegressionModel',
    'RandomForestClassifierModel', 
    'SVMModel',
    'NeuralNetworkClassifier',
    
    # Regression
    'LinearRegressionModel',
    'PolynomialRegressionModel',
    'RidgeRegressionModel',
    'LassoRegressionModel',
    
    # Clustering
    'KMeansModel',
    'DBSCANModel',
    'HierarchicalClusteringModel',
    
    # Deep Learning
    'CNNModel',
    'RNNModel',
    'LSTMModel',
]

__version__ = '1.0.0'
__author__ = 'Gabriel Demetrios Lafis'
