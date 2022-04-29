import random
import math 
import csv
import pandas as df
from sklearn.model_selection import train_test_split

#read and save contents of a csv file
with open('ciFile.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)    
    #filePandas = df.DataFrame(data)
    
#split data to training and test sets
#(pandas)
#training = data.sample(frac=0.75,random_state=25) #random state is a seed value
#test = data.drop(training.index)
#
#(sklearn)
training, test = train_test_split(data, test_size=0.25, random_state=25)

trainingInputs1 = []
trainingInputs2 = []
trainingOutputs = []
testInputs1 = []
testInputs2 = []
testOutputs = []

for x in training:
    trainingInputs1.append(float(x[0]))
    trainingInputs2.append(float(x[1]))
    trainingOutputs.append(float(x[2]))

for x in test:
    testInputs1.append(float(x[0]))
    testInputs2.append(float(x[1]))
    testOutputs.append(float(x[2]))

# weigths for inputs 
i1weights = [0, 0]
i2weights = [0, 0]
i1weights[0] = random.random()
i1weights[1] = random.random()
i2weights[0] = random.random()
i2weights[1] = random.random()

#weights for 2 hidden nodes
h1weight = random.random()
h2weight = random.random()

#when net reaches convergence or set no of epochs, then test the net using
#data from the test set, then study performance (output of test)

# set the learning rate
learningRate = 0.5

# list to store network outputs
networkOutputs = [0 for i in range (0,45)]

# array to store errors
errors = [0 for i in range (0,45)]

hasConvergence = False

euler = 2.7182

def noErrors(listOfErrors):
    noOfErrors = 0
    #iterate through error list
    #if there's at least one error, break the loop and return false later
    for x in listOfErrors:
        if x > 0.5 or x < -0.5:
            noOfErrors += 1
            
    print(noOfErrors)
    if noOfErrors == 0:
        return True
    else:
        return False

#TRAINING
while hasConvergence == False:
#for epoch in range (0, 200):
        #for i in range(0, len(training)):
    for i1,i2,target,Y,e in zip(trainingInputs1, trainingInputs2,
                                     trainingOutputs, networkOutputs,
                                     errors):
    #Forthpropagation
        '''
        i1 = float(trainingInputs1[i])
        i2 = float(trainingInputs2[i])
        target = float(trainingOutputs[i])
        '''
        
        f = i1 * i1weights[0]
        g = i2 * i2weights[0]
        k = i1 * i1weights[1]
        l = i2 * i2weights[1]

        sum1 = f + g
        sum2 = k + l
        #sigmoid transfer function - transform outputs of layer 1 into inputs of layer 2
        h1 = 1/(1 + euler **(-sum1))
        h2 = 1/(1 + euler **(-sum2))
        '''
        networkOutput[i] = (h1 * h1weight) + (h2 * h2weight)
        errors[i] = learningRate * (target - networkOutput[i])
        '''
        Y = (h1 * h1weight) + (h2 * h2weight)
        e = target - Y
    #Backpropagation
        
        #Hidden layer weights update:
        #how much error changes with respect to output
        '''
        a = -1 * (target - networkOutput[i])
        '''
        a = -1 * (target - Y)
        #how much final output changes with respect to the total inputs
        '''
        b = networkOutput[i] * (1 - networkOutput[i])
        '''
        b = Y * (1 - Y)
        #how much total input of the net changes with respect to hidden neuron1's weight
        '''
        c = 1 * networkOutput[i] * h1weight
        '''
        c = 1 * Y * h1weight
        #how much total input of the net changes with respect to hidden neuron2's weight
        '''
        d = 1 * networkOutput[i] * h2weight
        '''
        d = 1 * Y * h2weight
        h1weightUpdate = a * b * c
        h2weightUpdate = a * b * d
        h1weight = h1weight - h1weightUpdate
        h2weight = h2weight - h2weightUpdate
        
        #Layer1 weights update:
        #Related to h1
        #how much Y changes with respect to h1
        '''
        m = float(h1 * (networkOutput[i] - h1))
        '''
        m = h1 * (Y - h1)
        i1weightUpdate0 = a * b * m * f
        i2weightUpdate0 = a * b * m * g
        i1weights[0] = i1weights[0] - i1weightUpdate0
        i2weights[0] = i2weights[0] - i2weightUpdate0
        #Related to h2
        #how much Y changes with respect to h2
        '''
        j = float(h2 * (networkOutput[i] - h2))
        '''
        j = h2 * (Y - h2)
        i1weightUpdate1 = a * b * j * k
        i2weightUpdate1 = a * b * j * l
        i1weights[1] = i1weights[1] - i1weightUpdate1
        i2weights[1] = i2weights[1] - i2weightUpdate1
        #print(errors[i])
        print(i2weights)
    hasConvergence = noErrors(errors)

if hasConvergence == True:
        print("Training achieved convergence.")
        print("Presenting test set.")
