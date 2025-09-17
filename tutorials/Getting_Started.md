# Getting Started / Primeiros Passos

*A beginner's guide to the awesome-data-science-toolkit / Um guia para iniciantes do awesome-data-science-toolkit*

## Installation / Instalação

### English
To get started with this toolkit, you'll need to install the required dependencies:

```bash
# Install Python packages
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter

# Optional: Install additional packages for advanced features
pip install xgboost lightgbm optuna
```

### Português
Para começar com este toolkit, você precisará instalar as dependências necessárias:

```bash
# Instalar pacotes Python
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter

# Opcional: Instalar pacotes adicionais para funcionalidades avançadas
pip install xgboost lightgbm optuna
```

## Basic Imports / Importações Básicas

```python
# Data manipulation / Manipulação de dados
import pandas as pd
import numpy as np

# Machine Learning / Aprendizado de Máquina
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data visualization / Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Suppress warnings / Suprimir avisos
import warnings
warnings.filterwarnings('ignore')
```

## Data Loading and Cleaning / Carregamento e Limpeza de Dados

### English
Here's how to load and clean your dataset:

### Português
Aqui está como carregar e limpar seu conjunto de dados:

```python
# Load data / Carregar dados
# Replace 'your_data.csv' with your actual file path
# Substitua 'your_data.csv' pelo caminho real do seu arquivo
df = pd.read_csv('your_data.csv')

# Basic data exploration / Exploração básica dos dados
print("Dataset shape / Formato do dataset:", df.shape)
print("\nFirst 5 rows / Primeiras 5 linhas:")
print(df.head())

print("\nData types / Tipos de dados:")
print(df.dtypes)

print("\nMissing values / Valores ausentes:")
print(df.isnull().sum())

# Handle missing values / Tratar valores ausentes
# Option 1: Remove rows with missing values / Opção 1: Remover linhas com valores ausentes
df_clean = df.dropna()

# Option 2: Fill missing values / Opção 2: Preencher valores ausentes
# For numerical columns / Para colunas numéricas
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# For categorical columns / Para colunas categóricas
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Remove duplicates / Remover duplicatas
df = df.drop_duplicates()

print(f"\nCleaned dataset shape / Formato do dataset limpo: {df.shape}")
```

## Data Preparation / Preparação dos Dados

```python
# Encode categorical variables / Codificar variáveis categóricas
label_encoders = {}
for column in categorical_cols:
    if column in df.columns:  # Check if column still exists / Verificar se a coluna ainda existe
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Separate features and target / Separar características e variável alvo
# Assume the last column is the target / Assume que a última coluna é o alvo
X = df.iloc[:, :-1]  # Features / Características
y = df.iloc[:, -1]   # Target / Alvo

# Split data into training and testing sets / Dividir dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features / Escalar as características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape / Formato do conjunto de treino: {X_train_scaled.shape}")
print(f"Test set shape / Formato do conjunto de teste: {X_test_scaled.shape}")
```

## Basic Machine Learning / Aprendizado de Máquina Básico

### English
Let's train a simple Random Forest model:

### Português
Vamos treinar um modelo Random Forest simples:

```python
# Create and train the model / Criar e treinar o modelo
model = RandomForestClassifier(
    n_estimators=100,    # Number of trees / Número de árvores
    random_state=42,     # For reproducibility / Para reprodutibilidade
    max_depth=10,        # Maximum depth of trees / Profundidade máxima das árvores
    min_samples_split=5  # Minimum samples to split / Amostras mínimas para dividir
)

# Train the model / Treinar o modelo
model.fit(X_train_scaled, y_train)

# Make predictions / Fazer previsões
y_pred = model.predict(X_test_scaled)

# Evaluate the model / Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy / Acurácia do Modelo: {accuracy:.4f}")

print("\nDetailed Classification Report / Relatório Detalhado de Classificação:")
print(classification_report(y_test, y_pred))

# Feature importance / Importância das características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features / Top 10 Características Mais Importantes:")
print(feature_importance.head(10))
```

## Data Visualization / Visualização de Dados

### English
Create informative visualizations to understand your data:

### Português
Crie visualizações informativas para entender seus dados:

```python
# Set up the plotting style / Configurar o estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Distribution of target variable / Distribuição da variável alvo
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x=df.columns[-1])
plt.title('Target Variable Distribution / Distribuição da Variável Alvo')
plt.xlabel('Classes / Classes')
plt.ylabel('Count / Contagem')
plt.show()

# 2. Correlation heatmap / Mapa de calor de correlação
plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap / Mapa de Calor de Correlação das Características')
plt.tight_layout()
plt.show()

# 3. Feature importance visualization / Visualização da importância das características
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(10)
sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
plt.title('Top 10 Feature Importance / Top 10 Importância das Características')
plt.xlabel('Importance Score / Pontuação de Importância')
plt.ylabel('Features / Características')
plt.tight_layout()
plt.show()

# 4. Interactive plot with Plotly / Gráfico interativo com Plotly
fig = px.scatter_matrix(
    df.select_dtypes(include=[np.number]).sample(1000),  # Sample for performance
    title="Interactive Feature Scatter Matrix / Matriz de Dispersão Interativa das Características"
)
fig.show()

# 5. Model performance visualization / Visualização do desempenho do modelo
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix / Matriz de Confusão')
plt.xlabel('Predicted / Previsto')
plt.ylabel('Actual / Real')
plt.show()
```

## Advanced Tips / Dicas Avançadas

### English
- **Cross-validation**: Use `cross_val_score` for more robust model evaluation
- **Hyperparameter tuning**: Try `GridSearchCV` or `RandomizedSearchCV`
- **Feature engineering**: Create new features based on domain knowledge
- **Ensemble methods**: Combine multiple models for better performance

### Português
- **Validação cruzada**: Use `cross_val_score` para avaliação mais robusta do modelo
- **Ajuste de hiperparâmetros**: Experimente `GridSearchCV` ou `RandomizedSearchCV`
- **Engenharia de características**: Crie novas características baseadas no conhecimento do domínio
- **Métodos de ensemble**: Combine múltiplos modelos para melhor desempenho

```python
# Example of cross-validation / Exemplo de validação cruzada
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores / Pontuações de validação cruzada: {cv_scores}")
print(f"Mean CV accuracy / Acurácia média CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Example of hyperparameter tuning / Exemplo de ajuste de hiperparâmetros
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                          param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters / Melhores parâmetros: {grid_search.best_params_}")
print(f"Best cross-validation score / Melhor pontuação de validação cruzada: {grid_search.best_score_:.4f}")
```

## Next Steps / Próximos Passos

### English
1. Explore more advanced algorithms (XGBoost, LightGBM, Neural Networks)
2. Learn about feature selection techniques
3. Understand different evaluation metrics for your specific problem
4. Practice with different types of datasets (time series, text, images)
5. Learn about model deployment and monitoring

### Português
1. Explore algoritmos mais avançados (XGBoost, LightGBM, Redes Neurais)
2. Aprenda sobre técnicas de seleção de características
3. Entenda diferentes métricas de avaliação para seu problema específico
4. Pratique com diferentes tipos de conjuntos de dados (séries temporais, texto, imagens)
5. Aprenda sobre implantação e monitoramento de modelos

## Resources / Recursos

### English
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

### Português
- [Documentação Scikit-learn](https://scikit-learn.org/stable/)
- [Documentação Pandas](https://pandas.pydata.org/docs/)
- [Galeria Matplotlib](https://matplotlib.org/stable/gallery/index.html)
- [Tutorial Seaborn](https://seaborn.pydata.org/tutorial.html)

---

*Happy Data Science! / Feliz Ciência de Dados!* 🚀📊🤖
