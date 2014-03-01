import numpy as np
import nn, os, pickle
from pylab import * # naughty me...

''' These are a selection of routines that act as a wrapper around
training the neural network and visualizing the response. At every
iteration, output median difference between the solution and target
values. Every 20 iterations, test the current neural network on a few
tertiary test data sets and seperately record their mean
difference. These will be plotted to show the gradual improvement of
the neural network and to test for overfitting.
'''

def calculateInput(photometryData):
    # Takes a photometry class object, selects and normalizes a few
    # parameters and returns these as an array

    # First cull out obvious problem points
    sel =  (photometryData.type==1) & 
           (photometryData.flag1<=2) & 
           (photometryData.flag2<=2) & 
           (photometryData.sn1>=4.5) & 
           (photometryData.sn2>=4.5)

    type=(photometryData.type[sel]*-1.0)+2.0

    snr1=(photometryData.sn1[sel]-25.0)/50.0
    error1=(1.0-photometryData.flag1[sel])
    crowding1=(photometryData.crowd1[sel]*-5.0)+1.0
    sharpnessL1=20.0*(photometryData.sharp1[sel]+0.05)/3.0
    sharpnessU1=-4.0*(photometryData.sharp1[sel]-0.25)
    roundness1=(0.5-photometryData.round1[sel])/0.5

    snr2=(photometryData.sn2[sel]-25.0)/50.0
    error2=(1.0-photometryData.flag2[sel])
    crowding2=(photometryData.crowd2[sel]*-5.0)+1.0
    sharpnessL2=20.0*(photometryData.sharp2[sel]+0.05)/3.0
    sharpnessU2=-4.0*(photometryData.sharp2[sel]-0.25)
    roundness2=(0.5-photometryData.round2[sel])/0.5
    
    # These 6 parameters seem to be the most interesting
    return np.asfarray(zip(snr1, crowding1, sharpnessL1, snr2, crowding2, sharpnessL2))


def dataArray(photometryFileName):
    # Load a photometry object from a file and process the data
    with open(photometryFileName, 'rb') as photometryFile:
        print "Loading inputs"
        photometryData=pickle.load(photometryFile)
        return calculateInput(photometryData)


def wrapper():
    # Define some training sets and their target values
    training_set1 = {
                 'empty-field02-artificial.dat' : 0.9,
                 'empty-field02.dat' : 0.01,
                 'empty-field03.dat' : 0.01,
                 'empty-field04-artificial.dat' : 0.9,
                 'empty-field04.dat' : 0.01
                 }

    training_set2 = {'empty-field01-artificial.dat' : 0.9,
                 'empty-field01.dat' : 0.01
                 }

    training_set3 = {'empty-field05-artificial.dat' : 0.9,
                 'empty-field05.dat' : 0.01,
                 'empty-field06.dat' : 0.01
                 }

    training_set4 = {
                 'empty-field03-artificial.dat' : 0.9,
                 'empty-field07.dat' : 0.01,
                 'empty-field08.dat' : 0.01
                 }

    trainingSet1,trainingSet2,trainingSet3,trainingSet4=[],[],[],[]
    targets1,targets2,targets3,targets4=[],[],[],[]

    # For each training set, create an array of data and an array of
    # target values suitable for input into the neural network
    for name,t in training_set1.iteritems():
        arr = gi.dataArray(name)
        targets1 = np.append(targets1, np.zeros(np.shape(arr)[0])+t)
        trainingSet1.extend(arr)
    for name,t in training_set2.iteritems():
        arr = gi.dataArray(name)
        targets2 = np.append(targets2, np.zeros(np.shape(arr)[0])+t)
        trainingSet2.extend(arr)
    for name,t in training_set3.iteritems():
        arr = gi.dataArray(name)
        targets3 = np.append(targets3, np.zeros(np.shape(arr)[0])+t)
        trainingSet3.extend(arr)
    for name,t in training_set4.iteritems():
        arr = gi.dataArray(name)
        targets4 = np.append(targets4, np.zeros(np.shape(arr)[0])+t)
        trainingSet4.extend(arr)

    bestDiff=2
    # Create 20 random neural networks and select the best to train
    # on. This is an attempt to help push the solution to a global
    # minimum.
    for j in range(20):
        newNeuralNetwork = nn.NeuralNetwork(8)
        output = newNeuralNetwork.forwardPropagate(trainingSet1)
        diff = output-targets1
        mediandiff = np.median(np.abs(diff))  
        print j, mediandiff, np.min(diff), np.max(diff)
        if (mediandiff<bestDiff):
            print 'updating'
            bestDiff=mediandiff
            brain = newNeuralNetwork

    # Start Training the neural network
    if os.path.exists('brains/nn_output.dat'): os.system('rm brains/nn_output.dat')
    if os.path.exists('brains/intermediate_output.dat'): os.system('rm brains/intermediate_output.dat')
    with open('brains/intermediate_output.dat','wa') as outfile:
        for i in range(1000):
            # Every "maximumIterations" will backup the neural network and test it on some tertiary data.
            print 'starting prime iteration',i
            brain.trainNeuralNetwork(trainingSet1,targets1,maximumIterations=20,outputFile="brains/nn_output.dat")
            # Save copy of neural network in this intermediate state 
            with open('brains/nn_{:03d}.dat'.format(i),'wb') as out: 
                pickle.dump(brain,out)

            print 'analysing second dataset'
            output = brain.forwardPropagate(trainingSet2)
            diff = np.abs(output - targets2)
            mean1, median1, std1 = np.mean(diff), np.median(diff), np.std(diff) 

            print 'analysing third dataset'
            output = brain.forwardPropagate(trainingSet3)
            diff = np.abs(output - targets3)
            mean2, median2, std2 = np.mean(diff), np.median(diff), np.std(diff) 

            print 'analysing fourth dataset'
            output = brain.forwardPropagate(trainingSet4)
            diff = np.abs(output - targets4)
            mean3, median3, std3 = np.mean(diff), np.median(diff), np.std(diff) 

            # Write the results of the tests on the tertiary datasets.
            outfile.write('{0:03d}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}\n'.format(i, mean1,median1,std1, mean2,median2,std2, mean3,median3,std3)) 
            # Flush the file so we can break the loop if needed and still have data!
            outfile.flush()


def plotTrends():
    # Plot the progression of the solution as a function of iteration number and compare with tertiary dataset runs.
    dat=np.loadtxt('brains/nn_output.dat',delimiter=',')
    intdat=np.loadtxt('brains/intermediate_output.dat',delimiter=',')
    figure(1,figsize=(8,5))
    indices=np.arange(len(dat[:,0]))
    semilogy(indices,dat[:,2],'k-',lw=3,label='Training Set')
    plot((intdat[:,0]+1)*20-0.9,intdat[:,2],'-',color='r',ms=5,mec='r',label='Ancillary Test Set 1')
    plot((intdat[:,0]+1)*20-0,  intdat[:,5],'-',color='g',ms=5,mec='g',label='Ancillary Test Set 2')
    plot((intdat[:,0]+1)*20+0.9,intdat[:,8],'-',color='b',ms=5,mec='b',label='Ancillary Test Set 3')
    ylim([0.008,0.15])
    rcParams['legend.frameon']=False
    rcParams['legend.borderpad']=0.8
    legend(loc=3)
    xlabel('Iterations')
    text(-0.1,0.5,'Figure of Merit (target = 0)', rotation=90, va='center', transform=gca().transAxes)
    text(0.5,1.04,'Neural Network Training',fontsize=16,ha='center', transform=gca().transAxes)
    subplots_adjust(left=0.1,bottom=0.11,right=0.97,top=0.92)
    savefig('brains/nn_progression.png')


def plotResponse():
    # Plot the performance of the neural network in classifying data at defined iteration numbers.
    bins=np.linspace(0.,1.5,50)
    # Define which neural network states to test the output for. Note
    # that these are multiplied by 20 to determine the iteration state
    # of the network.
    values=[0,10,300,600,999]

    fig,ax = subplots(len(values),1,figsize=(7,10))
    dat1 = gi.dataArray('empty-field01-artificial.dat') 
    dat2 = gi.dataArray('empty-field01.dat')
    for i,number in enumerate(values):
        with open('brains/nn_{:03d}.dat'.format(number),'rb') as fn: 
            brain=pickle.load(fn)
        output = brain.forwardPropagate(dat1)
        ax[i].hist(output, bins=bins,histtype='step',color='b',lw=2,normed=True)
        output = brain.forwardPropagate(dat2)
        ax[i].hist(output, bins=bins,histtype='step',color='r',lw=2,normed=True)
        ax[i].axis([0.0,1.0,0,28])
        text(0.5,0.87,'Iteration {:d}'.format(20*number),ha='center',transform=ax[i].transAxes)
        if (i!=len(values)-1):
            ax[i].xaxis.set_major_locator(NullLocator())
        if i==0:
            text(0.68,0.87,'Background Galaxies',color='r',transform=ax[i].transAxes)
            text(0.68,0.77,'Stars',color='b',transform=ax[i].transAxes)
        ax[i].set_ylabel('N  (normalised)')
    xlabel('Neural Network Response')
    subplots_adjust(left=0.08,bottom=0.06,right=0.98,top=0.985,hspace=0.)
    savefig('network_response.png')


if __name__=='__main__':
    #wrapper()
    #plotTrends()
    plotResponse()
