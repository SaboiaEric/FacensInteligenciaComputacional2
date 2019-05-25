from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import itertools
import pandas as pd
import numpy as np
import nltk
import json
import pickle
import csv

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    resultado = modelo.fit(treino_dados, treino_marcacoes)
    fit_and_predict_score = str(modelo.score(treino_dados, treino_marcacoes))
    print("ETAPA TREINO "+ nome + "- Acurácia: " + fit_and_predict_score)
    #print("Devio Padrão: " + str(np.std(treino_dados,  dtype=np.float64)))
    print("==============================================================")
    return fit_and_predict_score

def valida_dados(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    metricas(resultado,validacao_marcacoes)

def metricas(resultado,validacao_marcacoes):
    
    #Apresentação do report contendo precision, recall and F-measures
    if len(set(resultado)) > 2:
        target_names = ['negativo', 'positivo', 'neutro']
        #os valores acima estão na ordem correta
        print(classification_report(validacao_marcacoes, resultado, target_names=target_names))
    else:
        print("Classification Report\n")
        target_names = ['negativo', 'positivo']
        #os valores acima estão na ordem correta
        print(classification_report(validacao_marcacoes, resultado, target_names=target_names))       

        #Apresentação dos true positive, false positive, true negative e false negative
        tn, fp, fn, tp = confusion_matrix(validacao_marcacoes, resultado).ravel() 
        msg = "True Positive: {0} \nFalse Positive: {1}\nTrue Negative: {2}\nFalse Negative: {3}".format(str(tn),str(fp),str(fn),str(tp))
        print(msg+"\n")

        matrix_confusao(validacao_marcacoes, resultado,target_names)

        print("Metrics\n")
        
        print("Precision: " + str(precision_score(validacao_marcacoes, resultado))) 
        print("Recall: " + str(recall_score(validacao_marcacoes, resultado))) 
        #Cria estatistica dos resultados.
        

    print("F-score: " + str(f1_score(validacao_marcacoes, resultado)))
    taxa_de_acerto_base = max(Counter(validacao_marcacoes).values()) * 100 / len(validacao_marcacoes)
    print("Taxa de acerto base: %f" % taxa_de_acerto_base)

    #Apresentação da accuracy e resultado final
    #print("Acurácia: " + str(accuracy_score(validacao_marcacoes, resultado)))

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Matrix de Confusão Normalizada')
    else:
        print('Matrix de Confusão Não Normalizada')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def matrix_confusao(validacao_marcacoes, resultado, target_names):
    cnf_matrix = confusion_matrix(validacao_marcacoes, resultado)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=target_names,title='Matrix de Confusão, sem normalização')

def cria_modelos(treino_dados, treino_marcacoes):
    resultados = {}
    
    # MultinomialNB
    modeloMultinomial = MultinomialNB(alpha=1.0, fit_prior=True)
    resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
    resultados[resultadoMultinomial] = modeloMultinomial
    
    # AdaBoostClassifier
    modeloAdaBoost = AdaBoostClassifier(random_state=0)
    resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
    resultados[resultadoAdaBoost] = modeloAdaBoost
    
    modeloDecisionTree = DecisionTreeClassifier(random_state=0)
    resultadoDecisionTree = fit_and_predict("DecisionTree", modeloDecisionTree, treino_dados, treino_marcacoes)
    resultados[resultadoDecisionTree] = modeloDecisionTree

    modeloRandomForest = RandomForestClassifier(random_state=0)
    resultadoRandomForest = fit_and_predict("RandomForest", modeloRandomForest, treino_dados, treino_marcacoes)
    resultados[resultadoRandomForest] = modeloRandomForest

    modeloLogisticRegression = LogisticRegression(random_state = 0)
    resultadoLogisticRegression = fit_and_predict("LogisticRegression", modeloLogisticRegression, treino_dados, treino_marcacoes)
    resultados[resultadoLogisticRegression] = modeloLogisticRegression

    #Verifica qual modelo teve a melhor perfomance
    maximo = max(resultados)
    return resultados[maximo], resultados

def divide_dados(dados):
    X = dados.Text.values
    y = dados.Sentimento.values 
    return train_test_split(X, y, test_size=0.2, random_state=42)

def pre_processamento(treino_dados, vectorizer):
    return vectorizer.fit_transform(treino_dados)

def pre_processamento_validacao(validacao_dados, vectorizer):
    return vectorizer.transform(validacao_dados)

def processamento_holdout(dados):
    print("\n***************************")
    print("PROCESSAMENTO - HOLDOUT")
    print("***************************")
    treino_dados, validacao_dados, treino_marcacoes, validacao_marcacoes = divide_dados(dados)

    vectorizer = CountVectorizer(analyzer = "word")
    bow_treino = pre_processamento(treino_dados, vectorizer)

    #Encontrando o modelo vencedor
    vencedor,resultados = cria_modelos(bow_treino, treino_marcacoes)
    print("Modelo vencedor: " + str(vencedor)+"\n")

    #Validando o modelo com novos dados
    vencedor.fit(bow_treino, treino_marcacoes)

    bow_validacao = pre_processamento_validacao(validacao_dados, vectorizer)
    valida_dados(vencedor, bow_validacao, validacao_marcacoes) 

    return resultados

def processamento_kfold(dados, marcacao):
    contador = 1
    print("\n***************************")
    print("PROCESSAMENTO - KFOLD")
    print("***************************\n")
    random_split = StratifiedShuffleSplit(marcacao, test_size=.25, random_state=0)
    for train_index, test_index in random_split.split(dados, marcacao):
        treino_dados, treino_marcacoes = dados[train_index], dados[test_index]
        validacao_dados, validacao_marcacoes = marcacao[train_index], marcacao[test_index]
        print("GRUPO: "+ str(contador))
        
        print(len(treino_dados))
        print(len(treino_marcacoes))
        print(len(validacao_dados))
        print(len(validacao_marcacoes))

        #Encontrando o modelo vencedor
        vencedor, resultados = cria_modelos(treino_dados, treino_marcacoes)
        print("Modelo vencedor: " + str(vencedor)+"\n")

        #Validando o modelo com novos dados
        vencedor.fit(treino_dados, treino_marcacoes)
        valida_dados(vencedor, validacao_dados, validacao_marcacoes) 

        contador+=1

def processamento_cross_val_score(atributos, rotulos, resultados):
    print("\n***************************\n")
    print("PROCESSAMENTO - KFOLD")
    print("\n***************************\n")
    contador = 1
    k = 5
    for iterator in resultados:
        print("GRUPO: "+ str(contador))
        print("Nome: "+ str(resultados[iterator]))        
        scores = cross_val_score(resultados[iterator], atributos, rotulos, cv=k, scoring='f1')
        taxa_de_acerto = np.mean(scores)
        print("Taxa de acerto: " + str(taxa_de_acerto))
        contador+=1

def processar(dados):
    resultados = processamento_holdout(dados)  
    print("\n***************************\n")
    print("FINALIZOU O HOLDOUT")
    print("\n***************************\n")
    '''
    vectorizer = CountVectorizer(analyzer = "word")
    atributos = pre_processamento(dados.Text.values,vectorizer)
    rotulos = pre_processamento_validacao(dados.Sentimento.values,vectorizer)
    processamento_cross_val_score(atributos, rotulos, resultados)
    '''

    
