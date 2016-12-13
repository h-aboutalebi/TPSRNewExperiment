from ConstructEnvironment import *
from TestGenerator import *
from environment import *

def main():
    start_time = time.time()
    print("Started...")
    Result2=[]
    Result1=[]
    Henkel1=[]
    Henkel2=[]
    environmen = environment([0, 1, 2, 3],[0, 1, 2, 3, 4, 5, 6, 7], rank=3)
    t=TestGenerator(3,environmen)
    testsdata =t.generate()
    File="logNewExperiment.txt" #for storing detail information
    deleteContent(File)
    File2="ResultNewExperiment.txt" #for storing results
    deleteContent(File2)
    with open(File, "a") as myfile:
        myfile.write("started .... " + '\n')
    with open(File2, "a") as myfile2:
        myfile2.write("started .... " + '\n')
    ObservationProbability=observation(Bias=5,Numberobservation=4)
    print(ObservationProbability)
# c=ConstructEnvironment(10,TestData=l,SizeOfTraining=100,stepSize=0.01,lambda1=10,numIteration=2,FileName=File)
    for step in [0.001,0.01,0.1,0.5,1]:
        for lamda1 in [10,3,1,0.1,0.01,0.005]:
            for i in [50,100,500,1000,3000]:  # size of Training data
                start_time = time.time()
                L1 = []
                L2 = []
                H1 = []
                H2 = []
                for k in range(4):
                    A = ConstructEnvironment(observationProbabilty=ObservationProbability,sizeOfCone=4, SizeOfTraining=i, TestData=testsdata, stepSize=step,
                                             lambda1=lamda1,
                                             numIteration=150, FileName=File)
                    L1.append(A.R1)
                    L2.append(A.R2)
                    H1.append(A.E1)
                    H2.append(A.E2)
                Result1.append(L1)
                Result2.append(L2)
                Henkel1.append(H1)
                Henkel2.append(H2)
            with open(File2, "a") as myfile2:
                myfile2.write('Results For step size equal to: ' + str(step) + ' and lamda1: ' + str(lamda1) + '\n')
                myfile2.write("Error Algorithm without Denoising On Test Data: " + str(Result1) + '\n')
                myfile2.write("Error Algorithm with Denoising On Test Data: " + str(Result2) + '\n')
                myfile2.write("Error Henkel Matrix of Algorithm without Denoising: " + str(Henkel1) + '\n')
                myfile2.write("Error Henkel Matrix of Algorithm with Denoising: " + str(Henkel2) + '\n')
                myfile2.write("Running time: "+ str(time.time() - start_time))
            print('Results For step size equal to: ', str(step), ' and lamda1: ', str(lamda1), '\n')
            print("Error Algorithm without Denoising On Test Data: ", Result1)
            print("Error Algorithm with Denoising On Test Data: ", Result2)
            print("Error Henkel Matrix of Algorithm without Denoising: ", Henkel1)
            print("Error Henkel Matrix of Algorithm with Denoising: ", Henkel2)
            print("Running Time : ", time.time() - start_time)
# TitleErrorTest = "Error Of Algorithms Applied on Test Data with step: " + str(step) + " and lambda1: " + str(lamda1)
# TitleErrorHenkel = "Error of Henkel Matrix with step: " + str(step) + " and lambda1: " + str(lamda1)
# g = Graph(TitleErrorTest, Result1, Result2, [50, 100, 1000, 4000, 10000])
# g = Graph(TitleErrorHenkel, Henkel1, Henkel2, [50, 100, 1000, 4000, 10000])


# this function gives the probabity distribution for observation.
# Note that the NumberObservation argument is half of the actual total number of observation as we have two type of different states with different action
def observation(Bias=5,Numberobservation=4):
    L=[]
    for i in range(Bias):
        A=np.random.dirichlet(np.ones(Numberobservation),size=1)[0]
        B=[]
        if(i!=0):
            for element in A:
                B.append(element)
            for k in range(Numberobservation):
                B.append(0)
        else:
            for k in range(Numberobservation):
                B.append(0)
            for element in A:
                B.append(element)
        L.append(B)
    return L












def deleteContent(fName):
    with open(fName, "w"):
        pass
if __name__ == '__main__':
    main()
