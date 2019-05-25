import pandas as pd
from  processamentoNovo import processar

'''
tweets = pd.read_excel('DataSet_Bolsonaro.xlsx', sheet_name='DS-PNNeu-CL', encoding='utf-8')
frases = tweets['Text'].str.lower()
print("Arquivo: DataSet_Bolsonaro.xlsx	-  Folha: DS-PNNeu-CL")
processar(tweets, frases)

tweets = pd.read_excel('DataSet_Bolsonaro.xlsx', sheet_name='DS-PNNeu-ID', encoding='utf-8')
frases = tweets['Text'].str.lower()
print("Arquivo: DataSet_Bolsonaro.xlsx	-  Folha: DS-PNNeu-ID")
processar(tweets, frases)

tweets = pd.read_excel('dados.xlsx', sheet_name='DS-PN-CL', encoding='utf-8')
frases = tweets['Text'].str.lower()
print("Arquivo: dados.xlsx	-  Folha: DS-PN-CL")
processar(tweets, frases)
'''

dataFrame = pd.read_excel('dados.xlsx', sheet_name='DS-PN-ID', encoding='utf-8')
print("Arquivo: dados.xlsx	-  Folha: DS-PN-ID")
processar(dataFrame)