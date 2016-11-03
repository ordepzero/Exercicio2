# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 04:20:45 2016

@author: PeDeNRiQue
"""

from pybrain.datasets  import ClassificationDataSet
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork


def convet(values):
    position = 0;
       
    for i in range(len(values)):
        if(values[i] > values[position]):
            position = i
    result = [0]*10
    result[position] = 1
    
    return result

def create_identity_matrix(x,y):
    if(x == y):
        return(1)
    else:
        return(0)
        
def isCorrect(matrix):
    
    for i in range(len(matrix)):
        temp = [0] * 10
        temp[i] = 1
        if(matrix[i] != temp):
            return False
    return True
    
def count_erros(matrix):
    errors = 0
    for i in range(10):
        for j in range(10):
            if((i == j and matrix[i][j] == 0) or (i != j and matrix[i][j] == 1)):
                errors += 1
                break
    return errors
    
def execute(learn_rate,momentum_rate,file_result):
    matrix = [] 
    
    for x in range(10):
        line = []
        for y in range(10):
            if(x == y):
                line.append(1)
            else:
                line.append(0)
            
        matrix.append(line)

    #print(matrix)

    train_data = ClassificationDataSet(10, 10,nb_classes=10)#TAMANHO DA ENTRADA, NUMERO DE CLASSES
    #test_data  = ClassificationDataSet(10, 10,nb_classes=10)

    #CRIANDO A BASE DE TREINAMENTO E DE TEST    
    for i in range(10):
        #print(matrix[i])
        train_data.addSample(matrix[i],matrix[i])
        #test_data.addSample(matrix[i],matrix[i])
        
        
    neuro = 5
    fnn = buildNetwork(train_data.indim, neuro, train_data.outdim)
    trainer = BackpropTrainer(fnn, train_data, learningrate=learn_rate,momentum=momentum_rate,verbose=False)
    
    epochs = 0
    for i in range(1000):    
        epochs += 1
        trainer.train()  
        
        matrix = []
        for i in range(10):
            r = convet(fnn.activate(train_data['input'][i]))
            matrix.append(r)
            #print(r)
        
        result = isCorrect(matrix)
        
        if(result):
            break
 
    matrix = []
    for i in range(10):
        r = convet(fnn.activate(train_data['input'][i]))
        matrix.append(r) 
        print(r)
    errors = count_erros(matrix)
 
    print ("Epocas: ",epochs)
    print ("Neurônio Entrada: ", train_data.indim)    
    print ("Neurônio Saída: ", train_data.outdim)
    print (trainer.testOnClassData())
    print (trainer.testOnData())
    print("Erros: ",errors)
    
    line_result = str(momentum_rate)+"\t"+str(learn_rate)+"\t"+str(errors/10)+"\t"+str(epochs)
    
    f.write(line_result+"\n")
    f.flush()
    
if __name__ == "__main__":
    
    f = open('resultados.txt', 'a')    
    f.write("momentum\tlearn_rate\tacuracia\tepochs\n")
    
    learn_rates = [0.25, 0.5, 0.75]
    momentums = [0.25, 0.5, 0.75, 0.9]
    
    
    for learn_rate in learn_rates:
        for momentum in momentums:   
            for i in range(10):
                execute(learn_rate,momentum,f)
    
    
    f.close()
    
    '''
    matrix = []
    for i in range(10):
        r = convet(fnn.activate(train_data['input'][i]))
        matrix.append(r)
        #print(r)
    
    isCorrect(matrix)
    '''