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

#log2(100) = 6.64

def create_identity_matrix(x,y):
    if(x == y):
        return(1)
    else:
        return(0)
        
def show_matrix(matrix):
    cont = 0
    for x in range(10):
        for y in range(10):
            if(matrix[cont] > 0.5):
                print(1, end="")
            else:
                print(0, end="")
            cont = cont + 1
        print()
        
if __name__ == "__main__":
    
    matrix = []
    data = []    
    
    for x in range(10):
        for y in range(10):
            matrix.append(create_identity_matrix(x,y))
    
    #matrix = [matrix]

    train_data = ClassificationDataSet(100, 100,nb_classes=100)#TAMANHO DA ENTRADA, NUMERO DE CLASSES
    test_data  = ClassificationDataSet(100, 100,nb_classes=100)

    #CRIANDO A BASE DE TREINAMENTO E DE TEST    
    train_data.addSample(matrix,matrix)
    test_data.addSample(matrix,matrix)
    '''
    print ("Number of training patterns: ", len(train_data))
    print ("Input and output dimensions: ", train_data.indim, train_data.outdim)
    print ("First sample (input, target, class):")
    print (test_data['input'], test_data['target'])
    '''
    #CRIANDO A REDE
    network = FeedForwardNetwork()
    inLayer = SigmoidLayer(train_data.indim)
    hiddenLayer = SigmoidLayer(7)
    outLayer = SigmoidLayer(train_data.outdim)
    
    #CRIANDO AS CAMADAS
    network.addInputModule(inLayer)
    network.addModule(hiddenLayer)
    network.addOutputModule(outLayer)
    
    #CONECTANDO AS CAMADAS
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)    
    
    #INSERINDO AS CONEXOES NA REDE
    network.addConnection(in_to_hidden)
    network.addConnection(hidden_to_out)
    
    network.sortModules()
    
    trainer = BackpropTrainer( network, dataset=train_data, momentum=0.5, verbose=True, weightdecay=0.25)
    
    for i in range(1):
        trainer.trainEpochs( 300)
    
    
    result = (network.activate(matrix)) 
    
    show_matrix(result)
    