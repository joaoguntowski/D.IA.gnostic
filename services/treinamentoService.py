# 1 - buscar os dados (x e y)
# 2 - separar os dados em treino (75%) e teste (25%) (X treino e Y treino - X teste e Y teste)
# 3 - treinar o modelo com os dados de treino
# 4 - avaliar o modelo com os dados de teste (acurácia)

from services.vetorizacaoService import vetorizacao, encode_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Importante: Usamos o Classifier para garantir que o save_model funcione corretamente depois
from xgboost import XGBClassifier as XGBoost 


# 1
def buscar_dados():
    x = vetorizacao()
    y = encode_y()
    return x, y

# 2
# separa os dados 
def separar_dados():
    
    # separar os dados em treino (80%) e teste (20%)
    # separar os dados em treino (75%) e teste (25%)
    # separar os dados em treino (70%) e teste (30%)
    
    # buscando os dados
    x, y = buscar_dados()
    
    # random_state=42 garante que a divisão seja sempre igual toda vez que rodar
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size=0.30, random_state=42)
    
    return X_treino, X_teste, Y_treino, Y_teste

# 3
# modelo é treinado com dados
def treinar_modelo():
    """
    ML - 
    Função para treinar o modelo de Machine Learning com o algoritmo XGBooster.

    Returns:
        O modelo treinado - que se chama HealthIA.
    """
    
    # Otimização: Chamamos separar_dados() uma vez e pegamos tudo o que precisamos
    # O _ (underline) serve para ignorar as variáveis de teste que não usaremos aqui
    X_treino, _, Y_treino, _ = separar_dados()
    
    # crio o modelo / instância
    # HealthIA = XGBoost()
    
    # Com Hiper Parâmetros:
    HealthIA = XGBoost(n_estimators=150, learning_rate=0.03, max_depth=4, min_child_weight=1, random_state=42)
    
    # treino o modelo
    HealthIA.fit(X_treino, Y_treino)
    
    return HealthIA

# 4
# me dá a acurácia de modelo treinado.
def acuracia_modelo():
    
    HealthIA = treinar_modelo()
    
    # Otimização: Chamamos separar_dados() e ignoramos o treino, pegando só o teste
    _, X_teste, _, Y_teste = separar_dados()
    
    # estalinha recebe as previsões do modelo me dando a resposta dele
    
    # predindo pro modelo treinar (só com teste)
    # e o modelo me dá a resposta
    Y_pred = HealthIA.predict(X_teste) 
    
    # aqui, estou batendo a resposta certa (que tenho de treino) com o que o modelo fez
    # gabarito 
    acuracia = accuracy_score(Y_teste, Y_pred)
    
    porcentagem = acuracia * 100
    
    return porcentagem

# se a acurácia for do meu agrado, dai salvo o modelo...