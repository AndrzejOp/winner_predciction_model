import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import shap

#Ładujemy dane
df = pd.read_csv('Challenger_Ranked_Games.csv')
df.drop('gameId', axis=1, inplace=True)
df.dropna(inplace=True)

#Wyrysowanie macierzy korelacji, aby wstępnie zobaczyć zależności cech (lepsze zrozumienie danych)
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f',  annot_kws={"size": 1})
plt.title('Macierz korelacji między cechami')
plt.xticks()
plt.yticks()
plt.show()


# Rozkład zmiennej docelowej, żeby mieć pewność, że jest po równo wygranych i przegranych (mniej więcej)
plt.figure(figsize=(6, 4))
sns.countplot(x='blueWins', data=df)
plt.title('Rozkład wyników dla wygranej niebieskich')
plt.show()


# Przygotowanie danych (standaryzacja i podział na zbiory)
X = df.drop(['blueWins', 'redWins'], axis=1)
y = df['blueWins']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = np.array(y)


# Tworzenie modelu
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = create_model()

early_stopping = EarlyStopping(patience=5, min_delta=0.001)
reduce_lr = ReduceLROnPlateau(patience=3, min_delta=0.001)

# Trening
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0,
          callbacks=[early_stopping, reduce_lr])

# Wyjaśnienie za pomocą SHAP, pomoże zrozumieć model, podjęte decyzję i wpływ konkretnych aspketów na wynik
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Wykres SHAP dla wpływu cech na predykcję (to o czym mowa wyżej)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.title('Wpływ cech na wynik')
plt.show()

# Macierz pomyłek i raport
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=['Przewidziano niebieskich', 'Przewidziano czerwonych'],
            yticklabels=['Naprawdę wygrali niebiescy', 'Naprawdę wygrali czerwoni'])
plt.title('Macierz pomyłek')
plt.ylabel('Prawdziwe wyniki')
plt.xlabel('Przewidziane wyniki')
plt.show()

print(classification_report(y_test, y_pred))


# Dodatkowe dane z pliku requirements.txt
# pandas~=2.2.3
# tensorflow~=2.18.0
# seaborn~=0.13.2
# matplotlib~=3.10.0
# numpy~=2.0.2
# shap~=0.46.0
# scikit-learn~=1.6.1
