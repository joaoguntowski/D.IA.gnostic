# 1 - Chamar o framework que tem o TF-IDF - ok
# 2 - instanciar o modelo de vetorização (tfidf = TfidfVectorizer()) (criar uma função) - ok
# 3 - pegar os dados e vetorizar - ok
# 4 - colocar em formato de números - ok

from sklearn.feature_extraction.text import TfidfVectorizer
from services.datasetService import dataset_completo
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import RSLPStemmer

# --- CONFIGURAÇÃO DO NLTK ---
# Baixa os recursos necessários na primeira execução
try:
    nltk.data.find('stemmers/rslp')
except LookupError:
    nltk.download('rslp')
    nltk.download('punkt') # Opcional, mas bom ter

def tratar_texto(texto):
    """
    Recebe um texto (ex: "dores de cabeça") e retorna as raízes (ex: "dor de cabec")
    """
    stemmer = RSLPStemmer()
    palavras = texto.split()
    # Reduz cada palavra à sua raiz
    palavras_raiz = [stemmer.stem(p) for p in palavras]
    return " ".join(palavras_raiz)


# Vetorizador (Usado para salvar o objeto .pkl)
def vetorizador():
    """
    Função que cria um vetorizador TF-IDF e o ajusta aos dados de entrada
    """
    tfidf = TfidfVectorizer()
    
    df = dataset_completo()
    
    # --- AQUI ESTAVA FALTANDO A MÁGICA ---
    # Pegamos os sintomas E aplicamos a função de tratamento
    X = df["sintomas"].astype(str).apply(tratar_texto)
    
    tfidf.fit(X)
    
    return tfidf

# Vetorizador vetorizando os dados (Usado para o treinamento imediato)
def vetorizacao():
    
    # instanciar
    tfidf = TfidfVectorizer()
    
    # dataset, pegando o X
    df = dataset_completo()
    
    # --- AQUI TAMBÉM PRECISA APLICAR ---
    x = df["sintomas"].astype(str).apply(tratar_texto)
    
    # treinar o modelo
    tfidf.fit(x)
    
    # criar a matrix / dados vetorizados
    X_tfidf = tfidf.transform(x)
    
    return X_tfidf

def encode_y():
    """
    Função para codificar a variável alvo (Y) usando label Encoding.
    """
    # instanciar essa classe
    label_encoder = LabelEncoder()
    # carregar o dataset
    df = dataset_completo()
    y = df["diagnostico"].astype(str)
    
    y_encoded = label_encoder.fit_transform(y)
    
    return y_encoded

def criar_encoder_objeto():
    """
    Cria e treina o LabelEncoder, retornando O OBJETO (a ferramenta),
    e não apenas os dados transformados.
    """
    label_encoder = LabelEncoder()
    df = dataset_completo()
    y = df["diagnostico"].astype(str)
    
    # Apenas FIT (Aprender), sem Transform (não queremos os números agora)
    label_encoder.fit(y)
    
    return label_encoder