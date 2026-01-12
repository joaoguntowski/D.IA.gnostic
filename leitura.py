from services.datasetService import dataset_completo
from services.vetorizacaoService import vetorizacao,encode_y
from services.treinamentoService import acuracia_modelo
from services.salvarmodeloService import salvar_encoderY, salvar_vetorizador, salvar_modelo

def salvar_modelo_healthIA():
    salvar_modelo()

def salvar_vetoriazador_healthIA():
    salvar_vetorizador()
    
def salvar_encoderY_healthIA():
    salvar_encoderY()

def prints_dataset():
    df = dataset_completo()
    #print(df)
    return df
    

def prints_vetorizacao():
    X_tfidf = vetorizacao()
    print(X_tfidf)
    

def prints_encodeY():
    y_encoded = encode_y()
    print(y_encoded)


def print_acuracia():
    acuracia = acuracia_modelo()
    print(f"Acurácia do modelo: {acuracia:.2f}%")


if __name__ == "__main__":
    
    # prints_dataset()
    
    # prints_vetorizacao()
    
    # df = prints_dataset()
    # print(df["diagnostico"].value_counts())
    
    # prints_encodeY()
    
    print_acuracia()
    
    #salvar_vetoriazador_healthIA()
    #salvar_encoderY_healthIA()
    #salvar_modelo_healthIA()
    