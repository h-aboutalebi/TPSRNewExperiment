from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import *
from environment import *
from states import *
from HmmState import *
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from NewExperimentDenoingAlgorithm import *


#0 stands for red observation, 1 stands for blue observation, 2 stands for green observation
#actions can be performed in this state [0,1,2,3] corresponding to [left,right,up, down]
class ConstructEnvironment:

#BiasIndex determine the distance between each two column in the torus. n determines the number of nodes.(The true number of nodes is n**2)
    def __init__(self, observationProbabilty,sizeOfCone,SizeOfTraining, TestData, stepSize, lambda1, numIteration,FileName,BiasIndex=5,n=31,ErrorInObservation=0.05):
        self.ErrorInObservation = ErrorInObservation
        self.DistributionObservation=observationProbabilty #Constains sublists which determines the distribution of observation for each state
        self.FileName = FileName
        self.SizeOfCone = sizeOfCone
        self.lambda1 = lambda1
        self.PathTrain=[[],[],[]]
        self.create(BiasIndex,n,SizeOfTraining, TestData, stepSize, numIteration)
        with open(self.FileName, "a") as myfile:
            myfile.write("Size Of Training: " + str(SizeOfTraining) + '\n')
            myfile.write("Rank: " + str(self.environment.rank) + '\n')
            myfile.write("stepsize: "+ str(stepSize)+" lambda1: "+ str(lambda1)+'\n')
            myfile.write("Average error Method without Denoising: " + str(self.R1) + '\n')
            myfile.write("Average error Method with Denoising: " + str(self.R2) + '\n')
            myfile.write("Average error Henkel Matrix without Denoising: " + str(self.E1) + '\n')
            myfile.write("Average error Henkel Matrix with Denoising: " + str(self.E2) + '\n')
        print("Size Of Training: ", SizeOfTraining)
        print("Average error Method without Denoising", self.R1)
        print("Average error Method with Denoising", self.R2)
        print("Average error Henkel Matrix without Denoising", self.E1)
        print("Average error Henkel Matrix with Denoising", self.E2)



#creates and draw the corresponding environment. we draw the figure patch by patch
    def create(self,BiasIndex,nodes,SizeOfTraining, TestData, stepSize, numIteration):
        Transition=[]
        for j in range(BiasIndex):
            Transition.append(0)
        HmmStates=[]
        #Here I am just constructing the corresponding HMM states for the environment
        for i in range(BiasIndex):
            Transition2 = copy.deepcopy(Transition)
            if (i==0):
                Transition2[Transition2.__len__()-1]=1
                A0=copy.deepcopy(Transition2)
                Transition2[Transition2.__len__()-1]=0
                Transition2[1]=1
                A1=copy.deepcopy(Transition2)
                Transition2[1]=0
                Transition2[0]=1
                A2=copy.deepcopy(Transition2)
                S=Hmmstate([A0,A1,A2,A2],self.DistributionObservation[i],[1/4,1/4,1/4,1/4],i)
                HmmStates.append(S)
            elif(i==BiasIndex-1):
                Transition2[i-1] = 1
                A0 = copy.deepcopy(Transition2)
                Transition2[i-1] = 0
                Transition2[0] = 1
                A1 = copy.deepcopy(Transition2)
                Transition2[1] = 0
                Transition2[0] = 1
                A2 = copy.deepcopy(Transition2)
                S = Hmmstate([A0, A1, A2, A2], self.DistributionObservation[i],[1/2,1/2], i)
                HmmStates.append(S)
            else:
                Transition2[i - 1] = 1
                A0 = copy.deepcopy(Transition2)
                Transition2[i - 1] = 0
                Transition2[i+1] = 1
                A1 = copy.deepcopy(Transition2)
                Transition2[i + 1] = 0
                Transition2[1] = 0
                Transition2[0] = 1
                A2 = copy.deepcopy(Transition2)
                S = Hmmstate([A0, A1, A2, A2], self.DistributionObservation[i],[1/2,1/2], i)
                HmmStates.append(S)
        self.environment=environment([0,1,2,3],[0, 1, 2, 3, 4, 5, 6, 7],rank=5)
        self.environment.HmmStates=HmmStates
        # plt.hold(True)
        SpaceColumn=BiasIndex
        theta = np.linspace(0, 2. * np.pi, nodes)
        phi = np.linspace(0, 2. * np.pi, nodes)
        theta, phi = np.meshgrid(theta, phi)
        c, a = 2, 1
        x = (c + a * np.cos(theta)) * np.cos(phi)
        y = (c + a * np.cos(theta)) * np.sin(phi)
        z = a * np.sin(theta)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111, projection='3d')
        # ax1.view_init(36, 26)
        # ax1.set_zlim(-2, 2)
        nodes-=1
        # ax1.scatter(x, y, z + 0.001, color='w', s=3, marker='o')
        #These two following lines are for testing:
        # ax1.scatter([x[0,0]], [y[0,0]], [z[0,0]], color='r', s=4, marker='o')
        # ax1.scatter([x[0,1]], [y[0,1]], [z[0,1]], color='g', s=4, marker='o')
        # ax1.scatter([x[0, 2]], [y[0, 2]], [z[0, 2]], color='y', s=4, marker='o')
        # ax1.scatter([x[0, 3]], [y[0, 3]], [z[0, 3]], color='b', s=4, marker='o')
        for i in range(nodes): #we draw the figure patch by patch
            S=[]
            if(i==nodes-1):
                iPlusOne=0
            else:
                iPlusOne=i+1
            for j in range(nodes):
                if (j == nodes - 1):
                    jPlusOne = 0
                else:
                    jPlusOne = j + 1
                X = [[x[i, j], x[i, jPlusOne]],[x[iPlusOne, j],x[iPlusOne, jPlusOne]]]
                Y = [[y[i, j], y[i, jPlusOne]], [y[iPlusOne, j], y[iPlusOne, jPlusOne]]]
                Z = [[z[i, j], z[i, jPlusOne]], [z[iPlusOne, j], z[iPlusOne, jPlusOne]]]
                colorpatch=self.getColorPatch(['red','blue','green'])
                if(colorpatch=='red'):
                    colorstate=0
                if (colorpatch == 'blue'):
                    colorstate = 1
                if (colorpatch == 'green'):
                    colorstate = 2
                s = states(index=[i,j],color=colorstate,position=[x[i, j],y[i, j],z[i, j]], actions=self.setAction(i,[0,1,2,3],BiasIndex))
                S.append(s)
                if(i!=0):
                    s.setLeftneighbor(self.environment.states[i-1][j])
                    self.environment.states[i-1][j].setRightneighbor(s)
                if(j!=0):
                    s.setDownneighbor(S[j-1])
                    S[j-1].setUpneighbor(s)
                if(j==nodes-1):
                    S[0].setDownneighbor(s)
                    s.setUpneighbor(S[0])
                if(i==nodes-1):
                    s.setRightneighbor(self.environment.states[0][j])
                    self.environment.states[0][j].setLeftneighbor(s)
                # ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, color=colorpatch, alpha=1, shade=True,linewidth=0)
                # ax1.scatter([x[i, j]], [y[i, j]], [z[i, j]], color='b', s=4, marker='o')

            self.environment.states.append(S)
        self.explore(SizeOfTraining,BiasIndex,nodes)
        # ax1.plot_wireframe(x, y, z, rstride=SpaceColumn, cstride=1, color='tan', alpha=0.7)
        # for l in range(self.PathTrain[0].__len__()):
        #     ax1.plot_wireframe(self.PathTrain[0][l], self.PathTrain[1][l], self.PathTrain[2][l], rstride=SpaceColumn, cstride=1, color='tan', alpha=0.5, linewidth=3)
        # plt.show()
        self.Exam(self.environment.TrainingSet, TestData, stepSize, numIteration,A=self.environment.StartingState)
        if (type(self.R1) == bool):
            if (self.R1 == False):
                with open(self.FileName, "a") as myfile:
                    myfile.write("**********Experiment Repeated********************Experiment Repeated********************Experiment Repeated********************Experiment Repeated**********")
                A = ConstructEnvironment(observationProbabilty=self.DistributionObservation,sizeOfCone=self.SizeOfCone, SizeOfTraining=SizeOfTraining, TestData=TestData,
                         stepSize=self.R2/1.4, numIteration=numIteration, FileName=self.FileName, lambda1=self.lambda1)
                self.R1, self.R2, self.E1, self.E2 =A.R1, A.R2, A.E1, A.E2




#function for robot exploring the environment
    def explore(self,TrajectoryLentgh,BiasIndex,nodes):
        Actions=[]
        for i in range(TrajectoryLentgh):
            List=[]
            List2 =[]
            x=[]
            y=[]
            z=[]
            (s1,s2)=self.randomStart(BiasIndex,nodes)
            HmmState=self.environment.HmmStates[s1%BiasIndex]
            self.environment.StartingState=copy.deepcopy(HmmState)
            S = self.environment.states[s1][s2]
            # print(i)
            for j in range(4):
                a=self.getAction(S)
                Actions.append(a)
                S=S.transitionToNextState(a)
                HmmState=HmmState.transToNextState(a,self.environment.HmmStates)
                observation=S.observation(self.environment, self.ErrorInObservation)
                List.append((a,observation))
                List2.append(a)
                List2.append(self.environment.observe(HmmState,1))
                x.append(S.position[0])
                y.append(S.position[1])
                z.append(S.position[2])
            self.PathTrain[0].append(x)
            self.PathTrain[1].append(y)
            self.PathTrain[2].append(z)
            self.environment.Trajectory.append(List)
            self.environment.TrainingSet.append(List2)
            # print(List2)
        Up=0
        Down=0
        Right=0
        Left=0
        for i in Actions:
            if(i==0):
                Left+=1
            elif(i==1):
                Right+=1
            elif(i==2):
                Up+=1
            elif(i==3):
                Down+=1
        print("Left Move:", Left/len(Actions))
        print("Right Move:", Right/len(Actions))
        print("Up Move:", Up / len(Actions))
        print("Down Move:", Down/len(Actions))


    # Construct matrices and calculate probabilities and produce results (Main Function)
    def Exam(self, TrainingSet, TestSet, stepSize, numIteration, A):
        RANK = self.environment.rank
        O = self.constructDynamicMatriices(TrainingSet)
        O_Modified = O[0]
        HenkelMatrix=O[0][:,0:33]
        U, S, R = np.linalg.svd(HenkelMatrix)
        print("Singular Values: ",S)
        R = R[0:RANK, :]
        b1 = HenkelMatrix[0,:]*R.T
        P_THR = np.linalg.pinv(HenkelMatrix*R.T)
        b_inf = P_THR*(HenkelMatrix[:,0])
        Result1 = []
        RealProbabilityResult=[]
        for i in TestSet:
            r1 = self.RealProbabilityTest(i, state=self.environment.StartingState)
            # if(r1==0):
            #     TestSet.remove(i)
            #     pass
            Result1.append('{:.10f}'.format(r1))
            # if(r1==0):
            #     r1=0.000000000000000001
            RealProbabilityResult.append(r1)
        # print(RealProbabilityResult)
        Result2 = []
        ResultWithoutDenoisng=[]
        j = 0
        for i in TestSet:
            r1=self.FakeprobabilityTesWithOutDenoising(i, O, P_THR, R, b_inf, b1)[0, 0]
            if(r1<0):
                r1 = 0
            ResultWithoutDenoisng.append('{:.10f}'.format(r1))
            Result2.append(abs(r1-RealProbabilityResult[j]))
            j += 1
        ListOfEqualEntities = self.FindEqualElementsOfDynamicMatrix(self.ExtendedTest, self.HISTORYelement)
        # print(ListOfEqualEntities)
        W = newAlgorithm(rank=RANK,  DynamicMatrix=O_Modified,weights=self.environment.ExtendedWeight, ListEqualEntitities=ListOfEqualEntities, lambda1=self.lambda1,step=stepSize, numberOfStep=numIteration,FileName=self.FileName)  # the hyperparameter of second constarints are adjusted here
        Q, R = W.GradientEngine()
        if (type(Q) == bool):
            if (Q == False):
                self.R1,self.R2= False,R
                return
        InverseR = R.T
        InverseR=InverseR[:33,:]
        NewP_TH = Q * R[:, :33]
        Newb1 = (NewP_TH[0])*InverseR
        NewP_THR = np.linalg.pinv(NewP_TH*InverseR)
        Newb_inf = NewP_THR*(NewP_TH[:,0])
        j = 0
        Result3 = []
        ResultDenoising=[]
        for i in TestSet:
            r1=self.FakeprobabilityTesWithDenoising(i, Q, R, NewP_THR, InverseR, Newb_inf, Newb1)[0, 0]
            if(r1<0):
                r1 = 0
            ResultDenoising.append('{:.10f}'.format(r1))
            Result3.append(abs(r1 - RealProbabilityResult[j]))
            j += 1
        print("Real Probability :")
        print(self.RealHenkelMatrix())
        print("Result Denoising: ")
        print(O[0][:,:33])
        print("Real Probability :")
        print(Result1)
        REALHenkel = self.RealHenkelMatrix() #Calculates the exact Henkel Matrix
        ErrorHenkelWithoutDenoing=np.linalg.norm(REALHenkel-O[0][:,:33],'fro')
        ErrorHenkelWithDenoing = np.linalg.norm(REALHenkel - NewP_TH, 'fro')
        AverageMethodWithoutDenoising = sum(Result2) / float(len(Result2))
        AverageMethodWithDenoising = sum(Result3) / float(len(Result3))
        self.R1, self.R2, self.E1, self.E2 =AverageMethodWithoutDenoising, AverageMethodWithDenoising,ErrorHenkelWithoutDenoing,ErrorHenkelWithDenoing

    # computes and returns an array containing all the observable matrices inculiding P(H,T),B(H,ao,T) for all ao
    def constructDynamicMatriices(self, TrainingSet):
        H=self.combination(self.SizeOfCone)
        T = self.combinationExtended(self.SizeOfCone)
        self.HISTORYelement = H
        self.TESTelemet = self.combination(self.SizeOfCone)
        self.ExtendedTest=T
        # print(T)
        self.environment.Weight = np.zeros((H.__len__(), self.ExtendedTest.__len__()))
        O1 = []
        for history in self.HISTORYelement:
            l = []
            for test in self.ExtendedTest:
                Output=self.ProbabilityTestGivenHistory(test, history, self.environment.indexMap[self.unpack(history)], self.environment.indexMap[self.unpack(test)], TrainingSet)
                l.append(Output[1])
                self.environment.Weight[self.environment.indexMap[self.unpack(history)],self.environment.indexMap[self.unpack(test)]]=Output[0]
            O1.append(l)
        self.environment.ExtendedWeight = np.matrix(self.environment.Weight)
        O2 = {}
        O1 = np.matrix(O1)
        for action in self.environment.actions:
            for observation in self.environment.observations:
                O3 = []
                l=np.matrix(O1[:,self.environment.indexMap[self.unpack([action,observation])]])
                for test in self.TESTelemet:
                    if (test.__len__()!=0):
                        l = np.hstack((l,O1[:,self.environment.indexMap[self.unpack([action,observation]+test)]]))
                O2[(action, observation)] = l
        R = []
        R.append(O1)
        R.append(O2)
        return R


    # calculates the joint probability of a test given its history and its corresponding weights P(A_1,Y_1,A_2,Y_2, ... ,Y_n) = P(Y_1|A_1)*P(Y_2|A_2,A_1,Y_1) *** it is implemented via dynamic programming for faster computational time ***
    def ProbabilityTestGivenHistory(self, Test, History, indexRow, indexColumn, TrainingSet):
        TEST = History + Test
        if self.unpack(TEST) in self.environment.PTH:
            # print('Failed')
            return (self.environment.PTH[self.unpack(TEST)][0],self.environment.PTH[self.unpack(TEST)][1])
        if (TEST.__len__() == 0):
            self.environment.PTH[self.unpack(TEST)] = [1,1]
            return (1,1)
        ProbabilityOfTest=1
        for i in range(1,TEST.__len__(),2):
            p=self.ComplementaryProbabilty(TEST[i],TEST[:i],TrainingSet)
            ProbabilityOfTest=ProbabilityOfTest*p
        HistoryCounter = 0
        for o in TrainingSet:
            if self.TwoGivenListIsEqual(TEST[0::2], o[0::2]):
                HistoryCounter += 1
        if (HistoryCounter == 0):
            self.environment.PTH[self.unpack(TEST)] = [0,0]
            return (0,0)
        # weight = HistoryCounter / len(TrainingSet)
        weight=HistoryCounter
        self.environment.PTH[self.unpack(TEST)] = [weight,ProbabilityOfTest]
        return (weight, ProbabilityOfTest)

    # calculates the true probability of sequence of action observation:
    # AO list is a sequence of action observation given in format [0,1,0,2,...] where odd digit is observation and even digits are actions
    def RealProbabilityTest(self, AOList, state):
        Sum = 0
        action = AOList[0]
        probableNextState = state.transition[action]
        j = 0
        Newlist = []
        for k in range(2, AOList.__len__()):
            Newlist.append(AOList[k])
        for i in probableNextState:
            if (i != 0):
                NextState = self.environment.HmmStates[j]
                # Decoder=baseN(AOList[1], 3)
                # if(len(Decoder)==1):
                #     Decoder='0'+Decoder
                # probabilityObservation = NextState.observationProbability[int(Decoder[0])]* NextState.observationProbability[int(Decoder[1])]
                probabilityObservation=NextState.observationProbability[AOList[1]]
                if (probabilityObservation != 0):
                    if (Newlist.__len__() != 0):
                        Sum = Sum + probabilityObservation * i * self.RealProbabilityTest(Newlist, NextState)
                    else:
                        Sum = Sum + probabilityObservation * i

            j += 1
        return Sum

    # calculates the true probability of sequence of action observation without applying denoising algorithm before:
    def FakeprobabilityTesWithOutDenoising(self, TEST, O, P_THR, R, b_inf, b1):
        Sum = P_THR*O[1][(TEST[0], TEST[1])]*R.T
        for i in range(2, TEST.__len__(), 2):
            Sum =Sum*P_THR*O[1][(TEST[i], TEST[i + 1])]*R.T
        return b1 * Sum * b_inf

    # calculates the true probability of sequence of action observation after applying denoising matrix before:
    def FakeprobabilityTesWithDenoising(self, TEST, Q, R, NewP_THR, InverseR, Newb_inf, Newb1):
        New_R = self.indexMatrix([TEST[0], TEST[1]], R)
        NewP_TaoH = Q * New_R
        Sum = NewP_THR * NewP_TaoH *InverseR
        for i in range(2, TEST.__len__(), 2):
            New_R = self.indexMatrix([TEST[i], TEST[i + 1]], R)
            NewP_TaoH = Q * New_R
            Sum = Sum* NewP_THR * NewP_TaoH *InverseR
        return Newb1 * Sum * Newb_inf

# This is a complementray function for def ProbabilityTestGivenHistory. It calculates P(Y_i|Y_1,A_1,Y_2, ..., A_i-1)
    def ComplementaryProbabilty(self,y,history,TrainingSet):
        Test=history+[y]
        if (self.unpack(Test) in self.environment.ConditionalPTH):
            return self.environment.ConditionalPTH[self.unpack(Test)]
        else:
            HistoryCounter = 0
            HistoryDenumenatorCounter = 0
            for data in TrainingSet:
                if(self.TwoGivenListIsEqual(Test,data)):
                    HistoryCounter+=1
                if(self.TwoGivenListIsEqual(history,data)):
                    HistoryDenumenatorCounter+=1
            if (HistoryDenumenatorCounter == 0):
                self.environment.ConditionalPTH[self.unpack(Test)]=0
                return 0
            else:
                self.environment.ConditionalPTH[self.unpack(Test)] = HistoryCounter/HistoryDenumenatorCounter
                return HistoryCounter/HistoryDenumenatorCounter

    # gives the Matrix corresponding to the index of starting and ending columns of P_TaoH in the fat matrix (Fat Matrix has extended test size compared to usual P_TH)
    def indexMatrix(self, ao: list, Matrix):
        if((ao[0],ao[1]) in self.environment.DenoisingPTH):
            return self.environment.DenoisingPTH[(ao[0],ao[1])]
        List = []
        INDEX = []
        TEST=self.TESTelemet
        for element in TEST:
            List.append(ao + element)
        for element in List:
            INDEX.append(self.environment.indexMap[self.unpack(element)])
        B_AO = np.matrix(Matrix[:, INDEX[0]])
        for index in INDEX[1:]:
            B_AO = np.hstack((B_AO, Matrix[:, index]))
        self.environment.DenoisingPTH[(ao[0], ao[1])]=B_AO
        return B_AO

    # create all the permutation of 0,1,2 of given size i
    def combination(self, i):
        T1 = [[]]
        T2 = []
        for i1 in self.environment.actions:
            for i2 in self.environment.observations:
                T1.append([i1, i2])
        return T1 + T2

    def combinationExtended(self, i):
        T1 = [[]]
        T2 = []
        T3=[]
        for i1 in self.environment.actions:
            for i2 in self.environment.observations:
                T1.append([i1, i2])
                for i3 in self.environment.actions:
                    for i4 in self.environment.observations:
                        T2.append([i1, i2, i3, i4])
        List=  T1 + T2
        for element in List:
            self.environment.indexMap[self.unpack(element)]=List.index(element)
        return T1 + T2

    def TwoGivenListIsEqual(self, L1, L2):
        for i in range(L1.__len__()):
            if (L1[i] != L2[i]):
                return False
        return True

    # construct the list for second constraints in denoising algorithm
    def FindEqualElementsOfDynamicMatrix(self, Test, History):
        MainList = []  # MainList consist of all combination Test+History where in each of its element first entity is the combination, second entity is the index of corresponding element in dynamic matrix
        for test in Test:
            for history in History:
                MainList.append([history + test, [self.environment.indexMap[self.unpack(history)], self.environment.indexMap[self.unpack(test)]]])
        ListOfEqualEntities = []
        while len(MainList) != 0:
            Flag, Q, MainList = self.FindEqualities(MainList[0], MainList)
            if (Flag):
                ListOfEqualEntities.append(Q)
        return ListOfEqualEntities

    # supporting function for FindEqualElementsOfDynamicMatrix
    def FindEqualities(self, i, List):
        Q = []
        Q.append(i[1])
        Flag = False
        List.remove(i)
        for j in List:
            if (j[0] == i[0]):
                Q.append(j[1])
        if len(Q) > 1:
            Flag = True
        for k in range(len(Q)):
            if (k != 0):
                List.remove([i[0], Q[k]])
        return Flag, Q, List

    #calculate the real probability of Henkel Matrix(Dynamic Matrix)
    def RealHenkelMatrix(self):
        HenkelReal=[]
        TEST=self.combination(12)
        for history in self.HISTORYelement:
            l=[]
            for test in TEST:
                if(len(history+test)==0):
                    l.append(1)
                else:
                    l.append(self.RealProbabilityTest(history+test,self.environment.StartingState))
            HenkelReal.append(l)
        return np.matrix(HenkelReal)


    # Given a list with elements produces a string containing the elements of the list
    def unpack(self, list):
        s = ''
        for l in list:
            s = s + str(l)
        return s


#Assigns color randomley to the patches based on the number of observation List is the list of colors
    def getColorPatch(self,List):
        r=random.uniform(0,1)
        l=len(List)
        k=1/l
        step=0
        for i in range(l):
            if(step<=r<k+step):
                return List[i]
            else:
                step+=k
        return List[l-1]

# checks whether based on the position of the given point it can turn just left, right or it can also goes up and downs.
    def setAction(self,i,Actions,BiasIndex):
        if(i%BiasIndex!=0):
            return [1/2,1/2]
        else:
            return [1/4,1/4,1/4,1/4]

#given state return the action of trajectory based on the possible valid action for that state
    def getAction(self,state: states):
        step=0
        r=random.uniform(0,1)
        for l in range(len(state.actions)):
            if(step<=r<step+state.actions[l]):
                return l
            else:
                step+=state.actions[l]
        return len(state.actions)-1

# Assigns the starting point of robot randomely
    def randomStart(self,BiasIndex,nodes):
        L=[]
        for n in range(1,nodes,BiasIndex):
                L.append(n)
        r=random.uniform(0,1)
        p=1/nodes
        step=0
        j1=0
        for i in range(nodes):
            if(step<=r<p+step):
                j1=i
                break
            else:
                step+=p
        i1=L[len(L)-1]
        step=0
        p=1/len(L)
        r = random.uniform(0, 1)
        for j in L:
            if (step <= r < p + step):
                i1=j
                break
            else:
                step+=p
        return (i1,j1)


#given a number and base, returns number in the given base
def baseN(num, b, numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])
