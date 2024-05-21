
import datetime as dt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib
import json
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df_ = pd.read_excel("datasets/veriler_hepsi_grupB.xlsx")
df = df_.copy()


def preprocess_data(df):
    # Fiyat(USD) sütununu oluşturma
    df["Fiyat(USD)"] = df["Fiyat(TL)"] / 32

    # Sayısal ve kategorik sütunları tanımlama
    num_cols = ["Dahili Hafıza (GB)", "Ekran Boyutu(inch)", "RAMG(GB)", "Kamera(MP)", "Ön Kamera(MP)"]
    cat_cols = ["Model", "Renk", "Garanti", "Kimden", "Durumu"]

    # XS sütununu düşürme

    df = df.drop(columns=["XS"], axis=1)

    # Pil(%) sütunundaki eksik değerleri medyan ile doldurma
    df["Pil(%)"] = df.groupby(["Model", "Durumu"])["Pil(%)"].transform(lambda x: x.fillna(x.median()))

    # log_Fiyat(USD) sütununu oluşturma
    df["log_Fiyat(USD)"] = np.log1p(df["Fiyat(USD)"])

    # Sayısal sütunları ölçekleme
    scaler = MinMaxScaler()
    scaled_num_cols = scaler.fit_transform(df[num_cols])
    scaled_num_cols = pd.DataFrame(scaled_num_cols, columns=num_cols)

    # Kategorik sütunları dönüştürme
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # Kategorik sütunları kodlama
    encoded_cols = pd.get_dummies(df[cat_cols], drop_first=True, dtype=int)

    # İşlenmiş verileri birleştirme
    merged_df = pd.concat([encoded_cols, scaled_num_cols], axis=1)

    return merged_df, df["log_Fiyat(USD)"]

X , y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge_model = Ridge()

# Hiperparametrelerin olası değerlerinin bir ızgarasını tanımlama
param_grid = {'alpha': [0.1, 1, 10, 100],
              'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
              'tol': [1e-4, 1e-3, 1e-2]}

# GridSearchCV oluşturma
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5)

# GridSearchCV ile modeli eğitme
grid_search.fit(X_train, y_train)



# En iyi modeli seçme
best_ridge_model = grid_search.best_estimator_

# Test seti üzerinde en iyi modelin performansını değerlendirme
score = best_ridge_model.score(X_test, y_test)
print(f"R^2 score on test set with best model: {score:.3f}")


# En iyi modeli kaydetme
joblib.dump(best_ridge_model, 'best_ridge_model.pkl')

# En iyi modelin hiperparametrelerini JSON dosyasına kaydetme
best_params_json = json.dumps(grid_search.best_params_)
with open('best_ridge_model_params.json', 'w') as f:
    f.write(best_params_json)


X_train.columns

X_test.columns