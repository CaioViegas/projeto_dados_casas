import pandas as pd
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def label_encoding(df, coluna):
    encoder = LabelEncoder()
    df[coluna] = encoder.fit_transform(df[coluna])
    return df

def target_encoding(df, coluna, alvo):
    encoder = TargetEncoder()
    df[coluna] = encoder.fit_transform(df[coluna], df[alvo])
    return df

def ordinal_encoding(df, coluna, mapping):
    encoder = OrdinalEncoder(categories=mapping)
    df[coluna] = encoder.fit_transform(df[[coluna]])
    return df

def binary_encoding(df, coluna):
    encoder = BinaryEncoder()
    df = encoder.fit_transform(df[[coluna]])
    return df

def hashing_encoding(df, coluna, n_bins=5):
    encoder = HashingEncoder(n_bins=n_bins)
    df = encoder.fit_transform(df[[coluna]])
    return df

def one_hot_encoding(df, coluna):
    encoder = OneHotEncoder(drop='first')  
    valores_codificados = encoder.fit_transform(df[[coluna]]).toarray()
    colunas_codificadas = encoder.get_feature_names_out([coluna])
    df_codificado = pd.DataFrame(valores_codificados, columns=colunas_codificadas, index=df.index)
    df = pd.concat([df, df_codificado], axis=1).drop(columns=[coluna])    
    return df

def aplicar_encoding(df, coluna, metodo_encoding, coluna_alvo=None, mapeamento_ordinal=None, n_bins=5):
    if metodo_encoding == 'label':
        df = label_encoding(df, coluna)
    elif metodo_encoding == 'onehot':
        df = one_hot_encoding(df, coluna)
    elif metodo_encoding == 'target' and coluna_alvo:
        df = target_encoding(df, coluna, coluna_alvo)
    elif metodo_encoding == 'ordinal' and mapeamento_ordinal:
        df = ordinal_encoding(df, coluna, mapeamento_ordinal[coluna])
    elif metodo_encoding == 'binary':
        df = binary_encoding(df, coluna)
    elif metodo_encoding == 'hashing':
        df = hashing_encoding(df, coluna, n_bins=n_bins)  
    return df

if __name__ == "__main__":
    df = pd.read_csv("./Data/casas_londres_traduzido.csv")
    df.drop(columns=["Endereco"], axis=1, inplace=True)
    colunas_target = ["Bairro", "Vista"]
    colunas_label = ["Jardim", "Garagem"]
    colunas_hot = ["Tipo Imovel", "Tipo Aquecimento", "Varanda", "Materiais", "Estilo Interior"]
    colunas_ordinal = ["Estado Predio"]

    for coluna in colunas_target:
        df = aplicar_encoding(df, coluna, metodo_encoding='target', coluna_alvo="Preco (£)")  

    for coluna in colunas_label:
        df = aplicar_encoding(df, coluna, metodo_encoding='label')

    for coluna in colunas_hot:
        df = aplicar_encoding(df, coluna, metodo_encoding='onehot')

    mapeamento_ordinal = {"Estado Predio": [["Old", "Renovated", "New"]]}
    for coluna in colunas_ordinal:
        df = aplicar_encoding(df, coluna, metodo_encoding='ordinal', mapeamento_ordinal=mapeamento_ordinal)
    
    df.to_csv("./Data/casas_londres_codificado.csv", index=False)