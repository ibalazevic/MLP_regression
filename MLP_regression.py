# authors: Ivana Balazevic & Franziska Horn
import random
import numpy as np
import matplotlib.pyplot as plt


def MLPBackpropagation(inputs, desiredOutputs, adaptLR=False):
    """
    perform backprob learning in a MLP
        inputs:
            - inputs: training data x values
            - desiredOutputs: training data y values
            - adaptLR: if learning rate should be adapted after each iteration
        outputs:
            - errorEvolution: vector with the error values for each iteration
            - finalOutputs: the output estimations to the input values using the trained MLP
    """
    # determine the number of neurons in each layer
    input_neurons = np.shape(inputs)[0]
    output_neurons = np.shape(desiredOutputs)[0]
    hidden_neurons = 3
    training_samples = np.shape(inputs)[1]
    # initialize learningrate and weights
    learningRate = 0.5    
    w_1 = np.random.rand(hidden_neurons,input_neurons+1)-0.5
    w_2 = np.random.rand(output_neurons,hidden_neurons)-0.5
    
    oldError = 1000000
    numOfIterations = 3000
    errorEvolution = np.zeros([numOfIterations,1])
    for n in xrange(numOfIterations):
        print "Iteration :", n
        # initialization for error backprop
        quadraticError = 0
        delta_1 = np.zeros([hidden_neurons, 1])
        delta_2 = np.zeros([output_neurons, 1])
        der_w_1 = np.zeros([hidden_neurons,input_neurons+1])
        der_w_2 = np.zeros([output_neurons,hidden_neurons])
        for i in xrange(training_samples):
            # calculate the output of the hidden units
            outputh = np.tanh(np.dot(w_1, np.concatenate(([[1]], [inputs[:,i]]),1).T)) # 3x2*(1x2)' = 3x1
            # calculate the output of the network
            networkOutput = np.dot(w_2, outputh) # 1x3 * 3x1 = 1x1
            # compute error
            quadraticError += 0.5 * (desiredOutputs[:,i] - networkOutput)**2
            delta_2 = networkOutput - desiredOutputs[:,i]
            delta_1 = (1-outputh**2)*(np.dot(w_2.T,delta_2))
            der_w_1 += np.dot(delta_1,np.concatenate(([[1]],[inputs[:,i]]),1))
            der_w_2 += np.dot(delta_2,outputh.T)
        # check if we're good enough    
        errorDiff = quadraticError-oldError
        # we are strict! also this is better for plotting...
        if abs(errorDiff/quadraticError) < 10**-9:
            break
        else:
            oldError = quadraticError
        if adaptLR:
            # 2.2
            learningRate = 1.02 * learningRate if errorDiff < 0 else 0.5 * learningRate
        
        # update weights
        w_1 = w_1 - (learningRate/training_samples)*der_w_1
        w_2 = w_2 - (learningRate/training_samples)*der_w_2
        print "Error: ", quadraticError
        errorEvolution[n,:] = quadraticError
    finalOutputs = np.zeros(np.shape(desiredOutputs))
    for i in xrange(training_samples):
        outputh = np.tanh(np.dot(w_1, np.concatenate(([[1]], [inputs[:,i]]),1).T))
        finalOutputs[:,i] = np.dot(w_2, outputh)
    return errorEvolution, finalOutputs
    
def plotErrorEvo(errorEvolution, errorEvolutionAdapt=0):
    """
    plot evolution of the error
        inputs:
            - errorEvolution: vector with error values for each iteration
            - errorEvolutionAdapt: same as above (with adaptive LR) if a second curve should be plotted in the figure
    """
    p1, = plt.plot(range(1,len(errorEvolution)+1), errorEvolution)
    if errorEvolutionAdapt.any():
        p2, = plt.plot(range(1,len(errorEvolutionAdapt)+1), errorEvolutionAdapt)
        plt.legend([p1,p2],['normal', 'adaptive LR'])
    plt.title('Error evolution')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Error')
    plt.savefig("errorevolution.eps")
    plt.show()
    plt.close()
    
def plotRegData(inputs, desiredOutputs, regresOutputs, regresOutputsAdapt=0):
    """
    plot regression results and original data
        inputs:
            - inputs: training data x values
            - desiredOutputs: training data y values
            - regresOutputs: outputs from the trained MLP
            - regresOutputsAdapt: second outputs with adaptive LR
    """
    p0, = plt.plot(np.linspace(0,1,100),np.sin(2*np.pi*np.linspace(0,1,100)),'k',lw=1)
    p1, = plt.plot(inputs[0],desiredOutputs[0], 'go')
    p2, = plt.plot(inputs[0],regresOutputs[0], 'ro')
    if regresOutputsAdapt.any():
        p3, = plt.plot(inputs[0],regresOutputsAdapt[0], 'bo')
        plt.legend([p0,p1,p2,p3],['sin(2pix)','training input', 'normal reg', 'adaptive LR'])
    else:
        plt.legend([p1,p2],['sin(2pix)','training input', 'normal reg'])
    plt.grid()
    #plt.axis('equal')
    plt.xlabel('input')
    plt.ylabel('output')
    plt.title('Regression')
    plt.savefig("data.eps")
    plt.show()
    plt.close()
        
if __name__ == '__main__':
    # initialization
    regressionData = open('RegressionData.txt', 'r').readlines()
    inputs = []
    desiredOutputs = []
    for i in regressionData:
        inputs.append(float(i.split()[0]))
        desiredOutputs.append(float(i.split()[1]))
    # make into arrays of size (1,x)
    inputs = np.array([inputs])
    desiredOutputs = np.array([desiredOutputs])
    # do awesome stuff    
    errorEvolution, regresOutputs = MLPBackpropagation(inputs, desiredOutputs)
    errorEvolutionAdapt, regresOutputsAdapt = MLPBackpropagation(inputs, desiredOutputs, adaptLR=True)
    plotErrorEvo(errorEvolution, errorEvolutionAdapt)
    plotRegData(inputs,desiredOutputs,regresOutputs,regresOutputsAdapt)
