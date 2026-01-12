
# Salvar o vetorizador
# Salvar o encoder - label (y)
# Salvar o modelo treinado

from services.vetorizacaoService import vetorizador, encode_y, criar_encoder_objeto
from services.treinamentoService import treinar_modelo
import pickle
import os

def salvar_vetorizador():
    """
    Salvar o vetorizador como arquivo no diretório model
    """
    vetorizador_criado = vetorizador()
    
    # salvar o vetorizador / wb = binário / dump = deixar em forma de arquivo
    caminho_arquivo = 'model/vetorizador_HealthIA.pkl'
    
    # Caso não tenha o diretório, criar...
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    
    with open(caminho_arquivo, "wb") as f:
        pickle.dump(vetorizador_criado, f)
        print("Vetorizador salvo com sucesso!")
        
def salvar_encoderY():
    """
    Salva o encoder Y como arquivo no diretório model.
    """
    encoderY_criado = encode_y()
    
    # salvar o encoder Y
    caminho_arquivo = 'model/encoderY_HealthIA.pkl'
    
    # Caso não tenha o diretório, criar...
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    
    with open(caminho_arquivo, "wb") as f:
        pickle.dump(encoderY_criado, f)
        print("Encoder Y salvo com sucesso!")
    

def salvar_modelo():
    """
    Salvar o modelo treinado como arquivo no diretório model.
    """
    HealthIA = treinar_modelo()
    
    # salvar o modelo
    caminho_arquivo = 'model/modelo_HealthIA.json'
    
    # Caso não tenha o diretório, criar...
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
 
    # --- CORREÇÃO DO ERRO ---
    # Usamos .get_booster() para pegar o núcleo do modelo e salvar direto,
    # ignorando a burocracia do Scikit-Learn que estava dando erro.
    HealthIA.get_booster().save_model(caminho_arquivo)
    
    print("Modelo salvo com sucesso!")
    
def salvar_encoderY():
    """
    Salva o encoder (O OBJETO) como arquivo no diretório model.
    """
    # CORREÇÃO: Usamos a função que retorna o objeto, não a lista de números
    encoder_objeto = criar_encoder_objeto()
    
    caminho_arquivo = 'model/encoderY_HealthIA.pkl'
    
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    
    with open(caminho_arquivo, "wb") as f:
        pickle.dump(encoder_objeto, f) # Agora estamos salvando a ferramenta certa!
        print("Encoder Y (Objeto) salvo com sucesso!")