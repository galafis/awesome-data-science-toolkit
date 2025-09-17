# 📚 Cartões de Referência Rápida | Quick Reference Cards

## 📖 Português

### 📁 Carregamento de Dados | Data Loading

| Função | Sintaxe | Exemplo | Descrição |
|--------|---------|---------|----------|
| Carregar CSV | `pd.read_csv(file)` | `df = pd.read_csv('data.csv')` | Carrega arquivo CSV |
| Carregar Excel | `pd.read_excel(file)` | `df = pd.read_excel('data.xlsx')` | Carrega arquivo Excel |
| Carregar JSON | `pd.read_json(file)` | `df = pd.read_json('data.json')` | Carrega arquivo JSON |
| Conectar BD | `pd.read_sql(query, conn)` | `df = pd.read_sql('SELECT * FROM table', conn)` | Carrega do banco de dados |
| Salvar CSV | `df.to_csv(file)` | `df.to_csv('output.csv', index=False)` | Salva como CSV |

### 🧹 Pré-processamento | Data Preprocessing

| Operação | Sintaxe | Exemplo | Descrição |
|----------|---------|---------|----------|
| Info básica | `df.info()` | `df.info()` | Informações sobre o dataset |
| Estatísticas | `df.describe()` | `df.describe()` | Estatísticas descritivas |
| Valores nulos | `df.isnull().sum()` | `df.isnull().sum()` | Conta valores nulos |
| Remover nulos | `df.dropna()` | `df.dropna(inplace=True)` | Remove linhas com nulos |
| Preencher nulos | `df.fillna(value)` | `df.fillna(df.mean())` | Preenche valores nulos |
| Duplicatas | `df.drop_duplicates()` | `df.drop_duplicates(inplace=True)` | Remove duplicatas |
| Normalização | `StandardScaler()` | `scaler.fit_transform(X)` | Normaliza dados |
| Encoding | `LabelEncoder()` | `encoder.fit_transform(y)` | Codifica categorias |

### 🤖 Machine Learning

| Algoritmo | Importação | Exemplo de Uso | Quando Usar |
|-----------|-------------|----------------|-------------|
| **Classificação** |
| Regressão Logística | `from sklearn.linear_model import LogisticRegression` | `model = LogisticRegression().fit(X, y)` | Classificação binária/multiclasse |
| Random Forest | `from sklearn.ensemble import RandomForestClassifier` | `model = RandomForestClassifier().fit(X, y)` | Dados complexos, interpretabilidade |
| SVM | `from sklearn.svm import SVC` | `model = SVC().fit(X, y)` | Dados não lineares |
| **Regressão** |
| Regressão Linear | `from sklearn.linear_model import LinearRegression` | `model = LinearRegression().fit(X, y)` | Relacionamentos lineares |
| Random Forest | `from sklearn.ensemble import RandomForestRegressor` | `model = RandomForestRegressor().fit(X, y)` | Dados complexos |
| **Clustering** |
| K-Means | `from sklearn.cluster import KMeans` | `model = KMeans(n_clusters=3).fit(X)` | Agrupamento de dados |

### 📊 Visualização | Visualization

| Gráfico | Biblioteca | Código | Uso |
|---------|------------|--------|-----|
| Linha | Matplotlib | `plt.plot(x, y)` | Tendências temporais |
| Barras | Seaborn | `sns.barplot(x='col1', y='col2', data=df)` | Comparação de categorias |
| Histograma | Matplotlib | `plt.hist(data)` | Distribuição de dados |
| Scatter | Seaborn | `sns.scatterplot(x='col1', y='col2', data=df)` | Correlação entre variáveis |
| Heatmap | Seaborn | `sns.heatmap(df.corr(), annot=True)` | Matriz de correlação |
| Boxplot | Seaborn | `sns.boxplot(x='category', y='value', data=df)` | Outliers e distribuição |

### 🔄 Pipeline Completo | Complete Workflow

```python
# 1. Carregamento
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Dados
df = pd.read_csv('data.csv')

# 3. Limpeza
df = df.dropna()  # Remove nulos
df = df.drop_duplicates()  # Remove duplicatas

# 4. Separação
X = df.drop('target', axis=1)
y = df['target']

# 5. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Predição
y_pred = model.predict(X_test_scaled)

# 9. Avaliação
print(f'Acurácia: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
```

---

## 🇺🇸 English

### 📁 Data Loading

| Function | Syntax | Example | Description |
|----------|--------|---------|-------------|
| Load CSV | `pd.read_csv(file)` | `df = pd.read_csv('data.csv')` | Load CSV file |
| Load Excel | `pd.read_excel(file)` | `df = pd.read_excel('data.xlsx')` | Load Excel file |
| Load JSON | `pd.read_json(file)` | `df = pd.read_json('data.json')` | Load JSON file |
| Database | `pd.read_sql(query, conn)` | `df = pd.read_sql('SELECT * FROM table', conn)` | Load from database |
| Save CSV | `df.to_csv(file)` | `df.to_csv('output.csv', index=False)` | Save as CSV |

### 🧹 Data Preprocessing

| Operation | Syntax | Example | Description |
|-----------|--------|---------|-------------|
| Basic info | `df.info()` | `df.info()` | Dataset information |
| Statistics | `df.describe()` | `df.describe()` | Descriptive statistics |
| Null values | `df.isnull().sum()` | `df.isnull().sum()` | Count null values |
| Drop nulls | `df.dropna()` | `df.dropna(inplace=True)` | Remove null rows |
| Fill nulls | `df.fillna(value)` | `df.fillna(df.mean())` | Fill null values |
| Duplicates | `df.drop_duplicates()` | `df.drop_duplicates(inplace=True)` | Remove duplicates |
| Scaling | `StandardScaler()` | `scaler.fit_transform(X)` | Normalize data |
| Encoding | `LabelEncoder()` | `encoder.fit_transform(y)` | Encode categories |

### 🤖 Machine Learning

| Algorithm | Import | Usage Example | When to Use |
|-----------|--------|---------------|-------------|
| **Classification** |
| Logistic Regression | `from sklearn.linear_model import LogisticRegression` | `model = LogisticRegression().fit(X, y)` | Binary/multiclass classification |
| Random Forest | `from sklearn.ensemble import RandomForestClassifier` | `model = RandomForestClassifier().fit(X, y)` | Complex data, interpretability |
| SVM | `from sklearn.svm import SVC` | `model = SVC().fit(X, y)` | Non-linear data |
| **Regression** |
| Linear Regression | `from sklearn.linear_model import LinearRegression` | `model = LinearRegression().fit(X, y)` | Linear relationships |
| Random Forest | `from sklearn.ensemble import RandomForestRegressor` | `model = RandomForestRegressor().fit(X, y)` | Complex data |
| **Clustering** |
| K-Means | `from sklearn.cluster import KMeans` | `model = KMeans(n_clusters=3).fit(X)` | Data grouping |

### 📊 Visualization

| Chart | Library | Code | Usage |
|-------|---------|------|-------|
| Line | Matplotlib | `plt.plot(x, y)` | Time trends |
| Bar | Seaborn | `sns.barplot(x='col1', y='col2', data=df)` | Category comparison |
| Histogram | Matplotlib | `plt.hist(data)` | Data distribution |
| Scatter | Seaborn | `sns.scatterplot(x='col1', y='col2', data=df)` | Variable correlation |
| Heatmap | Seaborn | `sns.heatmap(df.corr(), annot=True)` | Correlation matrix |
| Boxplot | Seaborn | `sns.boxplot(x='category', y='value', data=df)` | Outliers and distribution |

### 🔄 Complete Workflow

```python
# 1. Loading
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Data
df = pd.read_csv('data.csv')

# 3. Cleaning
df = df.dropna()  # Remove nulls
df = df.drop_duplicates()  # Remove duplicates

# 4. Feature separation
X = df.drop('target', axis=1)
y = df['target']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Prediction
y_pred = model.predict(X_test_scaled)

# 9. Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))
```

## 🚀 Utilitários do Toolkit | Toolkit Utilities

### 📦 Funções do Src

| Módulo | Função | Uso | Descrição |
|--------|--------|-----|----------|
| `src.preprocessing` | `clean_data(df)` | `cleaned_df = clean_data(df)` | Limpeza automática de dados |
| `src.preprocessing` | `handle_outliers(df)` | `df_no_outliers = handle_outliers(df)` | Remove outliers automático |
| `src.ml` | `quick_train(X, y)` | `model = quick_train(X, y)` | Treinamento rápido de modelo |
| `src.visualization` | `plot_correlations(df)` | `plot_correlations(df)` | Gráfico de correlações |
| `src.utils` | `memory_usage(df)` | `memory_usage(df)` | Uso de memória do dataset |

### 📈 Métricas de Avaliação | Evaluation Metrics

| Métrica | Código | Quando Usar |
|---------|--------|-------------|
| Acurácia | `accuracy_score(y_true, y_pred)` | Classificação balanceada |
| Precisão | `precision_score(y_true, y_pred)` | Minimizar falsos positivos |
| Recall | `recall_score(y_true, y_pred)` | Minimizar falsos negativos |
| F1-Score | `f1_score(y_true, y_pred)` | Balance entre precisão e recall |
| MAE | `mean_absolute_error(y_true, y_pred)` | Regressão (erros absolutos) |
| RMSE | `np.sqrt(mean_squared_error(y_true, y_pred))` | Regressão (penaliza erros grandes) |

---

*💡 Dica: Salve este arquivo como referência rápida para consultas durante seus projetos de ciência de dados!*

*💡 Tip: Save this file as a quick reference for lookups during your data science projects!*
