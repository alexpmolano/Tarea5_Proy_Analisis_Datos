#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Alexander Parra Molano


# In[6]:


# Importar librerías necesarias
import pandas as pd

# Cargar el dataset desde la carpeta local
file_path = 'C:\dataset\Titanic-Dataset.csv'
data = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset para verificar la carga
print(data.head())


# In[8]:


# Mostrar información general del dataset
print("Información del dataset:")
print(data.info())

# Estadísticas descriptivas de las columnas numéricas
print("\nEstadísticas descriptivas:")
print(data.describe())

# Mostrar los primeros 5 registros para inspección visual
print("\nPrimeros 5 registros:")
print(data.head())

# Verificar valores únicos en columnas categóricas (si existen)
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nValores únicos en la columna '{col}': {data[col].unique()}")

# Verificar valores faltantes
print("\nValores faltantes por columna:")
print(data.isnull().sum())


# In[16]:


from sklearn.preprocessing import StandardScaler, LabelEncoder

# Identificar columnas categóricas y numéricas
categorical_columns = data.select_dtypes(include=['object']).columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Tratamiento de valores faltantes
# Para columnas numéricas: rellenar con la media
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Para columnas categóricas: rellenar con la moda
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Codificación de variables categóricas
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Normalización de datos numéricos
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

print("\nDatos preprocesados:")
print(data.head())


# In[28]:


print(y.unique())


# In[30]:


# Convertir la variable objetivo a tipo entero
y = y.astype(int)


# In[32]:


from sklearn.ensemble import RandomForestRegressor

# Entrenar un modelo Random Forest para evaluación de características
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)


# In[34]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# Definir las características (X) y la variable objetivo (y)
X = data.drop('Survived', axis=1) 
y = data['Survived'] 

# Convertir la variable objetivo a tipo entero si es necesario
y = y.astype(int)

# Entrenar un modelo Random Forest para evaluar la importancia de las características
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

# Selección de características importantes
selector = SelectFromModel(rf, threshold="mean", max_features=5)  # Ajusta el umbral o el número de características
X_selected = selector.transform(X)

# Mostrar las características seleccionadas
selected_features = X.columns[selector.get_support()]
print("\nCaracterísticas seleccionadas:", selected_features)


# In[36]:


# Eliminar columnas no relevantes
X = X.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# Entrenar el modelo RandomForestRegressor
rf.fit(X, y)

# Selección de características importantes
selector = SelectFromModel(rf, threshold="mean", max_features=5)
X_selected = selector.transform(X)

# Mostrar las características seleccionadas
selected_features = X.columns[selector.get_support()]
print("\nCaracterísticas seleccionadas:", selected_features)


# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Seleccionar solo las características seleccionadas
X_selected = X[['Sex', 'Age', 'Fare']]

# Preprocesar la variable categórica 'Sex' (convertirla en numérica)
X_selected['Sex'] = X_selected['Sex'].map({'male': 0, 'female': 1})

# Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Entrenar el modelo
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predecir y evaluar el modelo
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[40]:


from sklearn.model_selection import GridSearchCV

# Definir los parámetros a ajustar
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Realizar una búsqueda en cuadrícula
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros
print("Mejores parámetros:", grid_search.best_params_)

# Usar el mejor modelo para hacer predicciones
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluar el modelo ajustado
print(classification_report(y_test, y_pred))


# In[41]:


X_selected.loc[:, 'Sex'] = X_selected['Sex'].map({'male': 0, 'female': 1})


# In[44]:


from sklearn.model_selection import GridSearchCV

# Definir los parámetros para GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Crear el modelo RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Ajustar el modelo usando GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

# Imprimir los mejores parámetros y el mejor score
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor puntuación de validación:", grid_search.best_score_)


# In[45]:


from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Predecir con el modelo ajustado
y_pred = grid_search.best_estimator_.predict(X)

# Generar la matriz de confusión
cm = confusion_matrix(y, y_pred)
print("Matriz de Confusión:")
print(cm)

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y, grid_search.best_estimator_.predict_proba(X)[:, 1])
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()


# In[48]:


from sklearn.model_selection import cross_val_score

# Realizar validación cruzada con el mejor modelo
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='accuracy')

# Imprimir los resultados de la validación cruzada
print(f"Puntuaciones de validación cruzada: {cv_scores}")
print(f"Puntuación media de validación cruzada: {cv_scores.mean()}")


# In[70]:


# Verificar las columnas de X_train y X_test
print("Columnas de X_train:", X_train.columns)
print("Columnas de X_test:", X_test.columns)


# In[72]:


# Aplicar la selección de características sobre X_train
selector = SelectFromModel(grid_search.best_estimator_, threshold="mean")
selector.fit(X_train, y_train)

# Verificar qué columnas fueron seleccionadas
selected_columns = X_train.columns[selector.get_support()]
print("Columnas seleccionadas:", selected_columns)


# In[78]:


X_test_selected = X_test[X_train.columns]

# Ahora aplica la transformación a X_test_selected
X_test_selected = selector.transform(X_test_selected)


# In[82]:


# Aplicar la selección de características al conjunto de entrenamiento
X_train_selected = selector.transform(X_train)

# Aplicar la misma transformación a X_test
X_test_selected = selector.transform(X_test)

# Verifica las dimensiones de ambos conjuntos de datos
print("Dimensiones de X_train_selected:", X_train_selected.shape)
print("Dimensiones de X_test_selected:", X_test_selected.shape)


# In[88]:


# Seleccionar solo las columnas 'Age' y 'Fare' tanto en X_train como en X_test
X_train_selected = X_train[['Age', 'Fare']]
X_test_selected = X_test[['Age', 'Fare']]

# Ajustar el modelo utilizando solo las características seleccionadas
grid_search.fit(X_train_selected, y_train)

# Realizar la predicción
y_test_pred = grid_search.best_estimator_.predict(X_test_selected)

# Calcular las métricas de evaluación
from sklearn.metrics import classification_report, accuracy_score
print("Métricas de evaluación:")
print(classification_report(y_test, y_test_pred))
print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")


# In[90]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Definir el rango de los hiperparámetros a explorar
param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árboles
    'max_depth': [10, 20, None],  # Profundidad de los árboles
    'min_samples_split': [2, 5, 10],  # Mínimas muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4],  # Mínimas muestras en las hojas
    'max_features': ['auto', 'sqrt', 'log2'],  # Número de características a considerar
    'bootstrap': [True, False]  # Usar muestreo bootstrap o no
}

# Crear el modelo
rf = RandomForestClassifier(random_state=42)

# Crear el objeto GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Ajustar el modelo con GridSearchCV
grid_search.fit(X_train_selected, y_train)

# Mostrar los mejores parámetros encontrados
print(f"Mejores hiperparámetros: {grid_search.best_params_}")


# In[91]:


# Realizar la predicción utilizando el mejor modelo encontrado
y_test_pred = grid_search.best_estimator_.predict(X_test_selected)

# Calcular las métricas de evaluación
from sklearn.metrics import classification_report, accuracy_score

# Mostrar el informe de clasificación
print(classification_report(y_test, y_test_pred))

# Calcular y mostrar la precisión
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")


# In[94]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paso 1: Predecir las etiquetas para X_test usando el mejor modelo
y_test_pred = grid_search.best_estimator_.predict(X_test_selected)

# Paso 2: Calcular las métricas de evaluación
print("Métricas de evaluación:")
print(classification_report(y_test, y_test_pred))

# Calcular la precisión
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")

# Paso 3: Visualizar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Crear un mapa de calor de la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"])
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()


# In[96]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_test_pred)

# Crear el gráfico de la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()


# In[98]:


from sklearn.metrics import roc_curve, auc

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, grid_search.best_estimator_.predict_proba(X_test_selected)[:,1])
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()


# In[100]:


# Obtener la importancia de las características
importances = grid_search.best_estimator_.feature_importances_

# Crear el gráfico de la importancia de las características
features = X_train_selected.columns
indices = importances.argsort()

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importancia')
plt.title('Importancia de las Características')
plt.show()


# In[ ]:




