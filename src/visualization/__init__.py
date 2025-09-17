# -*- coding: utf-8 -*-
"""
Visualization Module / Módulo de Visualização
============================================

This module provides visualization tools and utilities for exploratory data analysis
and statistical plotting. It includes functions for creating distribution plots,
statistical visualizations, and custom chart configurations.

Este módulo fornece ferramentas de visualização e utilitários para análise exploratória
de dados e plotagem estatística. Inclui funções para criar gráficos de distribuição,
visualizações estatísticas e configurações personalizadas de gráficos.

Functions:
    plot_feature_distribution: Plot the distribution of a feature from a DataFrame
    
Example:
    >>> import pandas as pd
    >>> from src.visualization import plot_feature_distribution
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> plot_feature_distribution(df, 'values', title='Value Distribution')

Author: Gabriel Demetrios Lafis
Version: 1.0.0
License: MIT
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple

# Configure matplotlib and seaborn defaults
plt.style.use('default')
sns.set_palette("husl")

__version__ = "1.0.0"
__author__ = "Gabriel Demetrios Lafis"
__email__ = "gabriel@example.com"

# Module exports
__all__ = [
    'plot_feature_distribution',
    'configure_plot_style',
    'save_plot'
]


def plot_feature_distribution(
    data: pd.DataFrame,
    feature: str,
    plot_type: str = 'histogram',
    title: Optional[str] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
    color: str = 'skyblue',
    alpha: float = 0.7,
    show_stats: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of a feature from a DataFrame.
    
    Plota a distribuição de uma feature de um DataFrame.
    
    This function creates various types of distribution plots including histograms,
    density plots, box plots, and violin plots. It automatically handles both
    numerical and categorical data with appropriate visualizations.
    
    Esta função cria vários tipos de gráficos de distribuição incluindo histogramas,
    gráficos de densidade, box plots e violin plots. Ela automaticamente lida com
    dados numéricos e categóricos com visualizações apropriadas.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing the data to plot.
        O DataFrame de entrada contendo os dados para plotar.
    
    feature : str
        The name of the column/feature to plot the distribution for.
        O nome da coluna/feature para plotar a distribuição.
    
    plot_type : str, default='histogram'
        Type of plot to create. Options: 'histogram', 'density', 'boxplot', 'violin'.
        Tipo de gráfico a criar. Opções: 'histogram', 'density', 'boxplot', 'violin'.
    
    title : str, optional
        Custom title for the plot. If None, auto-generates title.
        Título personalizado para o gráfico. Se None, gera título automaticamente.
    
    bins : int, default=30
        Number of bins for histogram (ignored for other plot types).
        Número de bins para histograma (ignorado para outros tipos de gráfico).
    
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
        Tamanho da figura como (largura, altura) em polegadas.
    
    color : str, default='skyblue'
        Color for the plot elements.
        Cor para os elementos do gráfico.
    
    alpha : float, default=0.7
        Transparency level for plot elements (0-1).
        Nível de transparência para elementos do gráfico (0-1).
    
    show_stats : bool, default=True
        Whether to display basic statistics on the plot.
        Se deve exibir estatísticas básicas no gráfico.
    
    save_path : str, optional
        Path to save the plot. If None, plot is not saved.
        Caminho para salvar o gráfico. Se None, gráfico não é salvo.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The matplotlib figure object containing the plot.
        O objeto figure do matplotlib contendo o gráfico.
    
    Raises:
    -------
    ValueError
        If the specified feature is not found in the DataFrame.
        Se a feature especificada não for encontrada no DataFrame.
    
    KeyError
        If the feature column doesn't exist in the DataFrame.
        Se a coluna da feature não existir no DataFrame.
    
    Examples:
    ---------
    >>> import pandas as pd
    >>> from src.visualization import plot_feature_distribution
    
    >>> # Create sample data / Criar dados de exemplo
    >>> df = pd.DataFrame({
    ...     'ages': np.random.normal(35, 10, 1000),
    ...     'category': np.random.choice(['A', 'B', 'C'], 1000)
    ... })
    
    >>> # Plot histogram / Plotar histograma
    >>> fig = plot_feature_distribution(df, 'ages', plot_type='histogram')
    
    >>> # Plot density / Plotar densidade
    >>> fig = plot_feature_distribution(df, 'ages', plot_type='density', color='red')
    
    >>> # Plot categorical data / Plotar dados categóricos
    >>> fig = plot_feature_distribution(df, 'category', plot_type='histogram')
    """
    
    # Input validation / Validação de entrada
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if feature not in data.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame columns: {list(data.columns)}")
    
    if data[feature].empty:
        raise ValueError(f"Feature '{feature}' contains no data")
    
    # Create figure and axis / Criar figura e eixo
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get feature data and remove NaN values / Obter dados da feature e remover valores NaN
    feature_data = data[feature].dropna()
    
    # Determine if data is numerical or categorical / Determinar se dados são numéricos ou categóricos
    is_numeric = pd.api.types.is_numeric_dtype(feature_data)
    
    # Set default title if not provided / Definir título padrão se não fornecido
    if title is None:
        title = f'Distribution of {feature}' if is_numeric else f'Count of {feature}'
    
    # Create the appropriate plot based on type and data / Criar gráfico apropriado baseado no tipo e dados
    if plot_type == 'histogram':
        if is_numeric:
            ax.hist(feature_data, bins=bins, color=color, alpha=alpha, edgecolor='black')
            ax.set_ylabel('Frequency / Frequência')
        else:
            # For categorical data, create a count plot
            value_counts = feature_data.value_counts()
            ax.bar(value_counts.index, value_counts.values, color=color, alpha=alpha)
            ax.set_ylabel('Count / Contagem')
            
    elif plot_type == 'density' and is_numeric:
        sns.histplot(feature_data, kde=True, stat='density', alpha=alpha, color=color, ax=ax)
        ax.set_ylabel('Density / Densidade')
        
    elif plot_type == 'boxplot':
        if is_numeric:
            ax.boxplot(feature_data, patch_artist=True, 
                      boxprops=dict(facecolor=color, alpha=alpha))
            ax.set_ylabel(feature)
        else:
            # For categorical data, show value counts as horizontal bars
            value_counts = feature_data.value_counts()
            ax.barh(value_counts.index, value_counts.values, color=color, alpha=alpha)
            ax.set_xlabel('Count / Contagem')
            
    elif plot_type == 'violin' and is_numeric:
        parts = ax.violinplot(feature_data, positions=[1], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(alpha)
        ax.set_ylabel(feature)
        ax.set_xticks([1])
        ax.set_xticklabels([feature])
        
    else:
        # Fallback to histogram for unsupported combinations
        if is_numeric:
            ax.hist(feature_data, bins=bins, color=color, alpha=alpha, edgecolor='black')
            ax.set_ylabel('Frequency / Frequência')
        else:
            value_counts = feature_data.value_counts()
            ax.bar(value_counts.index, value_counts.values, color=color, alpha=alpha)
            ax.set_ylabel('Count / Contagem')
    
    # Set labels and title / Definir rótulos e título
    ax.set_xlabel(feature)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add statistics if requested / Adicionar estatísticas se solicitado
    if show_stats and is_numeric:
        mean_val = feature_data.mean()
        std_val = feature_data.std()
        median_val = feature_data.median()
        
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}\nCount: {len(feature_data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    elif show_stats and not is_numeric:
        count = len(feature_data)
        unique_count = feature_data.nunique()
        most_common = feature_data.mode().iloc[0] if not feature_data.mode().empty else 'N/A'
        
        stats_text = f'Count: {count}\nUnique: {unique_count}\nMost Common: {most_common}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Improve layout / Melhorar layout
    plt.tight_layout()
    
    # Save plot if path provided / Salvar gráfico se caminho fornecido
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def configure_plot_style(style: str = 'seaborn', palette: str = 'husl') -> None:
    """
    Configure global plot styling options.
    
    Configura opções globais de estilo de gráfico.
    
    Parameters:
    -----------
    style : str, default='seaborn'
        Matplotlib style to use.
        Estilo matplotlib para usar.
    
    palette : str, default='husl'
        Seaborn color palette to use.
        Paleta de cores seaborn para usar.
    """
    if style in plt.style.available:
        plt.style.use(style)
    sns.set_palette(palette)


def save_plot(fig: plt.Figure, filepath: str, dpi: int = 300, format: str = 'png') -> None:
    """
    Save a matplotlib figure to file.
    
    Salva uma figura matplotlib para arquivo.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save.
        A figura para salvar.
    
    filepath : str
        Path where to save the figure.
        Caminho onde salvar a figura.
    
    dpi : int, default=300
        Resolution in dots per inch.
        Resolução em pontos por polegada.
    
    format : str, default='png'
        File format for saving.
        Formato de arquivo para salvar.
    """
    fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")


# Initialize default styling / Inicializar estilo padrão
configure_plot_style()
