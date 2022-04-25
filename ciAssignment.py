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
    trainingInputs1.append(x[0])
    trainingInputs2.append(x[1])
    trainingOutputs.append(x[2])

for x in test:
    testInputs1.append(x[0])
    testInputs2.append(x[1])
    testOutputs.append(x[2])

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
learningRate = 0.7

# array to store network outputs
networkOutput = []

# array to store errors
errors = []

hasConvergence = False

euler = 2.7182

sum1 = 0
sum2 = 0


def noErrors(listOfErrors):
    convergence = False
    #iterate through error list
    #if there's at least one error, break the loop and return false later
    for x in listOfErrors:
        if x < 0.5 and x > -0.5:
            convergence = True
        else:
            convergence = False
            break
    if convergence == True:
        return True
    else:
        return False


while hasConvergence == False:
    for i in range(0, len(trainingInputs1)):
    #Forthpropagation
        
        i1 = float(trainingInputs1[i])
        i2 = float(trainingInputs2[i])
        out = float(trainingOutputs[i])

        sum1 = (i1 * i1weights[0]) + (i2 * i2weights[0])
        sum2 = (i1 * i1weights[1]) + (i2 * i2weights[1])
            
        #sigmoid transfer function - transform outputs of layer 1 into inputs of layer 2
        h1 = 1/(1 + euler **(-sum1))
        h2 = 1/(1 + euler **(-sum2))
        
        networkOutput.insert(i, ((h1 * h1weight) + (h2 * h2weight)))
        errors.insert(i, learningRate * (out - networkOutput[i]))
    
    #Backpropagation
        
        #Hidden layer weights update:
        #how much error changes with respect to output
        a = -float(errors[i])
        #how much final output changes with respect to the total inputs
        b = float(networkOutput[i] * (1 - networkOutput[i]))
        #how much total input of the net changes with respect to hidden neuron1's weight
        c = float(networkOutput[i] * h1weight)
        #how much total input of the net changes with respect to hidden neuron2's weight
        d = float(networkOutput[i] * h2weight)
        h1weightUpdate = a * b * c
        h2weightUpdate = a * b * d
        h1weight = h1weight - h1weightUpdate
        h2weight = h2weight - h2weightUpdate
        
        #Layer1 weights update:
        #Related to h1
        #how much Y changes with respect to h1
        e = float(1 - h1)
        e2 = float(h1 * e)
        #total net input to h1 with respect to input1's weight1
        f = i1 * i1weights[0]
        #total net input to h1 with respect to input2's weight1
        g = i2 * i2weights[0]
        i1weightUpdate1 = a * b * e * f
        i2weightUpdate1 = a * b * e * g
        i1weights[0] = i1weights[0] - i1weightUpdate1
        i2weights[0] = i2weights[0] - i2weightUpdate1
        #Related to h2
        #how much Y changes with respect to h2
        j = float(1 - h2)
        j2 = float(h2 * j)
        #total net input to h1 with respect to input1's weight2
        k = i1 * i1weights[1]
        #total net input to h1 with respect to input2's weight2
        l = i2 * i2weights[1]
        i1weightUpdate2 = a * b * j2 * k
        i2weightUpdate2 = a * b * j2 * l
        i1weights[1] = i1weights[1] - i1weightUpdate2
        i2weights[1] = i2weights[1] - i2weightUpdate2
        print(errors[i])
    hasConvergence = noErrors(errors)

if hasConvergence == True:
        print("Achieved convergence.")
