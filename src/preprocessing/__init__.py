"""Data Preprocessing Module

Este módulo fornece funções utilitárias para pré-processamento de dados,
incluindo remoção de outliers e preenchimento de valores ausentes.

This module provides utility functions for data preprocessing,
including outlier removal and missing value imputation.

Author: Data Science Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from typing import Union, Literal


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers de uma coluna numérica usando o método IQR (Interquartile Range).
    
    Remove outliers from a numeric column using the IQR (Interquartile Range) method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        O DataFrame de entrada / Input DataFrame
    column : str
        Nome da coluna numérica para remoção de outliers / Name of numeric column for outlier removal
    method : str, default 'iqr'
        Método para remoção de outliers (atualmente apenas 'iqr' suportado) /
        Method for outlier removal (currently only 'iqr' supported)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com outliers removidos / DataFrame with outliers removed
    
    Raises:
    -------
    ValueError
        Se a coluna não existir ou não for numérica /
        If column doesn't exist or is not numeric
    
    Example:
    --------
    >>> import pandas as pd
    >>> data = {'values': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]}
    >>> df = pd.DataFrame(data)
    >>> clean_df = remove_outliers(df, 'values')
    >>> print(clean_df.shape[0])  # Outlier (100) removido / removed
    9
    """
    
    # Validação de entrada / Input validation
    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada no DataFrame / Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Coluna '{column}' deve ser numérica / Column '{column}' must be numeric")
    
    # Cálculo dos quartis / Calculate quartiles
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definição dos limites / Define boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtrar outliers / Filter outliers
    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    cleaned_df = df[mask].copy()
    
    print(f"Outliers removidos: {len(df) - len(cleaned_df)} registros")
    print(f"Outliers removed: {len(df) - len(cleaned_df)} records")
    
    return cleaned_df


def fill_missing_values(df: pd.DataFrame, column: str, 
                       method: Literal['mean', 'median', 'mode'] = 'mean') -> pd.DataFrame:
    """
    Preenche valores ausentes de uma coluna usando média, mediana ou moda.
    
    Fill missing values in a column using mean, median, or mode.
    
    Parameters:
    -----------
    df : pd.DataFrame
        O DataFrame de entrada / Input DataFrame
    column : str
        Nome da coluna para preenchimento / Name of column to fill
    method : {'mean', 'median', 'mode'}, default 'mean'
        Método de preenchimento / Filling method
        - 'mean': média aritmética (apenas colunas numéricas) / arithmetic mean (numeric columns only)
        - 'median': mediana (apenas colunas numéricas) / median (numeric columns only)
        - 'mode': moda (qualquer tipo de coluna) / mode (any column type)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame com valores ausentes preenchidos / DataFrame with missing values filled
    
    Raises:
    -------
    ValueError
        Se a coluna não existir ou método não for adequado para o tipo de dados /
        If column doesn't exist or method is not suitable for data type
    
    Examples:
    ---------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Exemplo com dados numéricos / Example with numeric data
    >>> data = {'scores': [85, 90, np.nan, 88, 92, np.nan, 87]}
    >>> df = pd.DataFrame(data)
    >>> filled_df = fill_missing_values(df, 'scores', method='mean')
    >>> print(filled_df['scores'].isna().sum())  # 0 valores ausentes / 0 missing values
    0
    >>> 
    >>> # Exemplo com dados categóricos / Example with categorical data
    >>> data = {'category': ['A', 'B', np.nan, 'A', 'C', np.nan, 'A']}
    >>> df = pd.DataFrame(data)
    >>> filled_df = fill_missing_values(df, 'category', method='mode')
    >>> print(filled_df['category'].isna().sum())  # 0 valores ausentes / 0 missing values
    0
    """
    
    # Validação de entrada / Input validation
    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada no DataFrame / Column '{column}' not found in DataFrame")
    
    # Criar cópia para não modificar o DataFrame original / Create copy to avoid modifying original
    df_filled = df.copy()
    
    # Verificar se há valores ausentes / Check if there are missing values
    missing_count = df_filled[column].isna().sum()
    if missing_count == 0:
        print(f"Nenhum valor ausente encontrado na coluna '{column}'")
        print(f"No missing values found in column '{column}'")
        return df_filled
    
    # Aplicar método de preenchimento / Apply filling method
    if method == 'mean':
        if not pd.api.types.is_numeric_dtype(df_filled[column]):
            raise ValueError(f"Método 'mean' requer coluna numérica / Method 'mean' requires numeric column")
        fill_value = df_filled[column].mean()
        
    elif method == 'median':
        if not pd.api.types.is_numeric_dtype(df_filled[column]):
            raise ValueError(f"Método 'median' requer coluna numérica / Method 'median' requires numeric column")
        fill_value = df_filled[column].median()
        
    elif method == 'mode':
        mode_values = df_filled[column].mode()
        if len(mode_values) == 0:
            raise ValueError(f"Não foi possível calcular a moda para a coluna '{column}' / Could not calculate mode for column '{column}'")
        fill_value = mode_values.iloc[0]  # Usar a primeira moda se houver múltiplas / Use first mode if multiple
        
    else:
        raise ValueError(f"Método '{method}' não suportado. Use 'mean', 'median' ou 'mode' / Method '{method}' not supported. Use 'mean', 'median', or 'mode'")
    
    # Preencher valores ausentes / Fill missing values
    df_filled[column].fillna(fill_value, inplace=True)
    
    print(f"Preenchidos {missing_count} valores ausentes na coluna '{column}' usando {method}: {fill_value}")
    print(f"Filled {missing_count} missing values in column '{column}' using {method}: {fill_value}")
    
    return df_filled


# Exemplo de uso / Usage example
if __name__ == "__main__":
    # Criar dados de exemplo / Create sample data
    np.random.seed(42)
    sample_data = {
        'numeric_col': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, np.nan, np.nan],  # Com outlier e valores ausentes
        'category_col': ['A', 'B', 'A', 'C', 'A', 'B', np.nan, 'A', 'C', np.nan, 'A', 'B']
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("\nDados originais / Original data:")
    print(df_sample)
    print(f"\nValores ausentes por coluna / Missing values per column:")
    print(df_sample.isna().sum())
    
    # Demonstração das funções / Function demonstration
    print("\n" + "="*50)
    print("DEMONSTRAÇÃO DAS FUNÇÕES / FUNCTION DEMONSTRATION")
    print("="*50)
    
    # 1. Preenchimento de valores ausentes / Fill missing values
    print("\n1. Preenchendo valores ausentes / Filling missing values:")
    df_filled = fill_missing_values(df_sample, 'numeric_col', method='mean')
    df_filled = fill_missing_values(df_filled, 'category_col', method='mode')
    
    # 2. Remoção de outliers / Remove outliers
    print("\n2. Removendo outliers / Removing outliers:")
    df_clean = remove_outliers(df_filled, 'numeric_col')
    
    print("\nDados finais / Final data:")
    print(df_clean)
