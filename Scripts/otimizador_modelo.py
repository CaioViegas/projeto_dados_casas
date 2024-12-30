import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

def preparar_dados(df, coluna_alvo):
    X = df.drop(columns=[coluna_alvo], axis=1)
    y = df[coluna_alvo]
    return X, y

def criar_pipeline():
    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('gb', GradientBoostingRegressor(random_state=101))
    ])
    return pipeline

def otimizar_modelo(X_train, y_train, pipeline, parametros):
    grid_search = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    return grid_search.best_estimator_

def calcular_metricas(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return mse, mae, r2, rmse

def executar_pipeline(df, coluna_alvo, parametros, caminho_salvar=None):
    X, y = preparar_dados(df, coluna_alvo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    scaler = MaxAbsScaler()
    pipeline = criar_pipeline()
    modelo_otimizado = otimizar_modelo(X_train, y_train, pipeline=pipeline, parametros=parametros)
    cv_scores = cross_val_score(modelo_otimizado, X_train, y_train, cv=5, scoring='r2')
    print(f"Pontuação de Cross-Validation (R2): {cv_scores}")
    y_pred = modelo_otimizado.predict(X_test)
    mse, mae, r2, rmse = calcular_metricas(y_test, y_pred)
    resultados = {
        'MSE': mse,
        'MAE': mae,
        'R²': r2,
        'RMSE': rmse
    }
    df_resultados = pd.DataFrame([resultados])
    print("\nResultados do modelo:")
    print(df_resultados)

    if caminho_salvar:
        joblib.dump(modelo_otimizado, caminho_salvar)
        print(f"\nModelo salvo em {caminho_salvar}")

if __name__ == "__main__":
    param_grid = {
        'gb__n_estimators': [50, 100, 200],  
        'gb__learning_rate': [0.01, 0.1, 0.2],  
        'gb__max_depth': [3, 5, 7],  
    }
    df = pd.read_csv("./Data/casas_londres_codificado.csv")
    caminho_salvar = "./Modelo/gradient_boosting_otimizado.joblib"
    executar_pipeline(df, "Preco (£)", param_grid, caminho_salvar=caminho_salvar)