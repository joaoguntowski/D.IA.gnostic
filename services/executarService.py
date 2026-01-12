import os
import joblib
import xgboost as xgb
import numpy as np

# IMPORTANTE: Importar a função de tratamento que criamos no outro arquivo
from services.vetorizacaoService import tratar_texto 

# Classe com os métodos (funções) para executar o modelo treinado
class DiagnosticoIA:
    
    def __init__(self, caminhoModelo: str):
        self.caminho_raiz = caminhoModelo
        self.HealthIA = xgb.XGBClassifier()
        
        # Carrega o modelo JSON (núcleo)
        self.HealthIA.load_model(os.path.join(self.caminho_raiz, 'modelo_HealthIA.json'))
        
        # Carrega as ferramentas auxiliares
        self.vetorizadortfidf = joblib.load(os.path.join(self.caminho_raiz, 'vetorizador_HealthIA.pkl'))
        self.encoderYPronto = joblib.load(os.path.join(self.caminho_raiz, 'encoderY_HealthIA.pkl'))
        

    def predict_simples(self, sintomas):
        # 1. Garantir que seja uma string única
        if isinstance(sintomas, list):
            sintomas_string = " ".join(sintomas)
        else:
            sintomas_string = sintomas
            
        # --- NOVO: REGRA DE MÍNIMO DE PALAVRAS ---
        # Se o usuário digitar menos de 3 palavras, nem tenta prever.
        # Ex: "dor cabeca" (2 palavras) -> Bloqueia
        if len(sintomas_string.split()) < 3:
             return ["Pouca informação. Por favor, digite pelo menos 3 sintomas ou detalhes."]

        sintomas_tratados = tratar_texto(sintomas_string)
        
        # DEBUG
        print(f"--- DIAGNÓSTICO ---")
        print(f"Texto: {sintomas_tratados}")
        
        sintomas_vetorizados = self.vetorizadortfidf.transform([sintomas_tratados])
        
        probabilidades = self.HealthIA.predict_proba(sintomas_vetorizados)
        maior_certeza = np.max(probabilidades)
        indice_vencedor = np.argmax(probabilidades)
        
        nome_diagnostico = self.encoderYPronto.inverse_transform([indice_vencedor])
        
        print(f"Vencedor: {nome_diagnostico[0]}")
        print(f"Certeza: {maior_certeza * 100:.2f}%")
        print(f"-------------------")
        
        # Baixar a régua para aceitar o diagnóstico de 46%
        LIMITE_DE_CONFIANCA = 0.35
        
        if maior_certeza < LIMITE_DE_CONFIANCA:
            # Retorna uma lista com 2 itens: [Titulo, Detalhe]
            return ["Inconclusivo", f"Certeza baixa: {maior_certeza*100:.1f}%"]

        else:
            nome_cru = nome_diagnostico[0]
            nome_bonito = nome_cru.replace("_", " ").title()
    
            # AQUI ESTÁ A MÁGICA:
            # Item 0 (Doença): "Gripe"
            # Item 1 (Probabilidade): "64,3%" (Formatado bonito)
            return [nome_bonito, f"{maior_certeza*100:.1f}%"]
        