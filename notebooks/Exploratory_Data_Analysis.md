# Exploratory Data Analysis / Análise Exploratória de Dados

## English Version

### 1. Introduction
This guide provides a comprehensive step-by-step approach to Exploratory Data Analysis (EDA), designed for professional use and easy conversion to Jupyter notebooks.

### 2. Data Loading
```python
# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization import plot_distribution, correlation_heatmap, categorical_analysis
from src.preprocessing import clean_data, handle_missing_values, detect_outliers

# Load dataset
def load_data(file_path):
    """
    Load data from various formats
    Args:
        file_path (str): Path to the data file
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Example usage
# df = load_data('data/dataset.csv')
```

### 3. Data Overview
```python
def data_overview(df):
    """
    Provide comprehensive overview of the dataset
    """
    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n=== BASIC STATISTICS ===")
    print(df.describe(include='all'))
    
    print("\n=== UNIQUE VALUES ===")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")

# Example usage
# data_overview(df)
```

### 4. Data Visualization
```python
def comprehensive_visualization(df):
    """
    Create comprehensive visualizations using src.visualization module
    """
    # Numerical variables distribution
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plot_distribution(df, col, title=f'Distribution of {col}')
    
    # Correlation analysis
    if len(numerical_cols) > 1:
        correlation_heatmap(df[numerical_cols], title='Correlation Matrix')
    
    # Categorical variables analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        categorical_analysis(df, col, title=f'Analysis of {col}')
    
    # Outlier detection visualization
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df.boxplot(column=col)
        plt.title(f'Boxplot: {col}')
        
        plt.subplot(1, 2, 2)
        df[col].hist(bins=30)
        plt.title(f'Histogram: {col}')
        plt.tight_layout()
        plt.show()

# Example usage
# comprehensive_visualization(df)
```

### 5. Data Cleaning
```python
def clean_dataset(df):
    """
    Comprehensive data cleaning using src.preprocessing module
    """
    print("=== STARTING DATA CLEANING ===")
    
    # Handle missing values
    df_cleaned = handle_missing_values(df, strategy='auto')
    
    # Detect and handle outliers
    outliers = detect_outliers(df_cleaned, method='iqr')
    print(f"Outliers detected: {len(outliers)} rows")
    
    # Clean data (remove duplicates, standardize formats)
    df_final = clean_data(df_cleaned, 
                         remove_duplicates=True,
                         standardize_text=True,
                         convert_dtypes=True)
    
    print(f"Cleaning completed: {df.shape[0]} -> {df_final.shape[0]} rows")
    return df_final

# Example usage
# df_clean = clean_dataset(df)
```

### 6. Conversion Tips for Jupyter Notebook
```python
# Cell 1: Setup and Imports
# Copy the import statements above

# Cell 2: Data Loading
# Use load_data function with your specific file path

# Cell 3: Initial Overview
# Run data_overview function

# Cell 4-N: Visualizations
# Run visualization functions one by one
# Add markdown cells with insights between code cells

# Final Cell: Cleaned Dataset
# Save the cleaned dataset
# df_clean.to_csv('cleaned_dataset.csv', index=False)
```

---

## Versão em Português

### 1. Apresentação
Este guia oferece uma abordagem abrangente e passo a passo para Análise Exploratória de Dados (AED), projetado para uso profissional e fácil conversão para notebooks Jupyter.

### 2. Carregamento de Dados
```python
# Importar bibliotecas essenciais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization import plot_distribution, correlation_heatmap, categorical_analysis
from src.preprocessing import clean_data, handle_missing_values, detect_outliers

# Carregar conjunto de dados
def carregar_dados(caminho_arquivo):
    """
    Carrega dados de vários formatos
    Args:
        caminho_arquivo (str): Caminho para o arquivo de dados
    Returns:
        pd.DataFrame: Conjunto de dados carregado
    """
    if caminho_arquivo.endswith('.csv'):
        df = pd.read_csv(caminho_arquivo)
    elif caminho_arquivo.endswith('.xlsx'):
        df = pd.read_excel(caminho_arquivo)
    elif caminho_arquivo.endswith('.json'):
        df = pd.read_json(caminho_arquivo)
    else:
        raise ValueError("Formato de arquivo não suportado")
    
    print(f"Dataset carregado com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df

# Exemplo de uso
# df = carregar_dados('data/dataset.csv')
```

### 3. Visão Geral dos Dados
```python
def visao_geral_dados(df):
    """
    Fornece visão geral abrangente do conjunto de dados
    """
    print("=== VISÃO GERAL DO DATASET ===")
    print(f"Formato: {df.shape}")
    print(f"Uso de memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== TIPOS DE DADOS ===")
    print(df.dtypes)
    
    print("\n=== VALORES AUSENTES ===")
    ausentes = df.isnull().sum()
    percentual_ausentes = (ausentes / len(df)) * 100
    df_ausentes = pd.DataFrame({
        'Contagem Ausentes': ausentes,
        'Percentual': percentual_ausentes
    })
    print(df_ausentes[df_ausentes['Contagem Ausentes'] > 0])
    
    print("\n=== ESTATÍSTICAS BÁSICAS ===")
    print(df.describe(include='all'))
    
    print("\n=== VALORES ÚNICOS ===")
    for col in df.columns:
        contagem_unica = df[col].nunique()
        print(f"{col}: {contagem_unica} valores únicos")

# Exemplo de uso
# visao_geral_dados(df)
```

### 4. Visualização de Dados
```python
def visualizacao_abrangente(df):
    """
    Cria visualizações abrangentes usando o módulo src.visualization
    """
    # Distribuição de variáveis numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    for col in colunas_numericas:
        plot_distribution(df, col, title=f'Distribuição de {col}')
    
    # Análise de correlação
    if len(colunas_numericas) > 1:
        correlation_heatmap(df[colunas_numericas], title='Matriz de Correlação')
    
    # Análise de variáveis categóricas
    colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns
    for col in colunas_categoricas:
        categorical_analysis(df, col, title=f'Análise de {col}')
    
    # Visualização de detecção de outliers
    for col in colunas_numericas:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df.boxplot(column=col)
        plt.title(f'Boxplot: {col}')
        
        plt.subplot(1, 2, 2)
        df[col].hist(bins=30)
        plt.title(f'Histograma: {col}')
        plt.tight_layout()
        plt.show()

# Exemplo de uso
# visualizacao_abrangente(df)
```

### 5. Limpeza de Dados
```python
def limpar_dataset(df):
    """
    Limpeza abrangente de dados usando o módulo src.preprocessing
    """
    print("=== INICIANDO LIMPEZA DOS DADOS ===")
    
    # Tratar valores ausentes
    df_limpo = handle_missing_values(df, strategy='auto')
    
    # Detectar e tratar outliers
    outliers = detect_outliers(df_limpo, method='iqr')
    print(f"Outliers detectados: {len(outliers)} linhas")
    
    # Limpar dados (remover duplicatas, padronizar formatos)
    df_final = clean_data(df_limpo, 
                         remove_duplicates=True,
                         standardize_text=True,
                         convert_dtypes=True)
    
    print(f"Limpeza concluída: {df.shape[0]} -> {df_final.shape[0]} linhas")
    return df_final

# Exemplo de uso
# df_limpo = limpar_dataset(df)
```

### 6. Dicas para Conversão em Notebook Jupyter
```python
# Célula 1: Configuração e Importações
# Copie as declarações de import acima

# Célula 2: Carregamento de Dados
# Use a função carregar_dados com o caminho específico do seu arquivo

# Célula 3: Visão Geral Inicial
# Execute a função visao_geral_dados

# Célula 4-N: Visualizações
# Execute as funções de visualização uma por uma
# Adicione células markdown com insights entre as células de código

# Célula Final: Dataset Limpo
# Salve o dataset limpo
# df_limpo.to_csv('dataset_limpo.csv', index=False)
```

## Professional Usage Notes / Notas para Uso Profissional

### English:
- Always document your findings in markdown cells when converting to Jupyter
- Use version control for your analysis notebooks
- Include data source documentation and metadata
- Consider creating automated reports using this structure
- Validate assumptions with domain experts

### Português:
- Sempre documente suas descobertas em células markdown ao converter para Jupyter
- Use controle de versão para seus notebooks de análise
- Inclua documentação da fonte de dados e metadados
- Considere criar relatórios automatizados usando esta estrutura
- Valide suposições com especialistas do domínio

## Next Steps / Próximos Passos

1. **Feature Engineering**: Create new variables based on insights
2. **Statistical Testing**: Validate hypotheses with appropriate tests
3. **Model Preparation**: Prepare data for machine learning models
4. **Reporting**: Create executive summaries and visualizations

1. **Engenharia de Características**: Crie novas variáveis baseadas em insights
2. **Testes Estatísticos**: Valide hipóteses com testes apropriados
3. **Preparação de Modelo**: Prepare dados para modelos de machine learning
4. **Relatórios**: Crie resumos executivos e visualizações
