#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import *
#from environment import *
#from states import *
#import matplotlib.pyplot as plt
#import numpy as np
#import random
#from NewExperimentDenoingAlgorithm import *
#
#
##0 stands for red observation, 1 stands for blue observation, 2 stands for green observation
#
#class ConstructEnvironment:
#
##BiasIndex determine the distance between each two column in the torus. n determines the number of nodes.(The true number of nodes is n**2)
#    def __init__(self,SizeOfTraining, TestData, stepSize, lambda1, numIteration,FileName,BiasIndex=5,n=30,ErrorInObservation=0.05):
#        self.ErrorInObservation = ErrorInObservation
#        self.FileName = FileName
#        self.lambda1 = lambda1
#        self.PathTrain=[[],[],[]]
#        self.create(BiasIndex,n,SizeOfTraining, TestData, stepSize, numIteration)
#
#
##creates and draw the corresponding environment. we draw the figure patch by patch
#    def create(self,BiasIndex,nodes,SizeOfTraining, TestData, stepSize, numIteration):
#        self.environment=environment([0,1,2,3],[[0,0],[0,1],[0,1],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]],rank=2)
#        plt.hold(True)
#        SpaceColumn=BiasIndex
#        theta = np.linspace(0, 2. * np.pi, nodes)
#        phi = np.linspace(0, 2. * np.pi, nodes)
#        theta, phi = np.meshgrid(theta, phi)
#        c, a = 2, 1
#        x = (c + a * np.cos(theta)) * np.cos(phi)
#        y = (c + a * np.cos(theta)) * np.sin(phi)
#        z = a * np.sin(theta)
#        fig = plt.figure()
#        ax1 = fig.add_subplot(111, projection='3d')
#        ax1.view_init(36, 26)
#        ax1.set_zlim(-2, 2)
#        nodes-=1
#        # ax1.scatter(x, y, z + 0.001, color='w', s=3, marker='o')
#        #These two following lines are for testing:
#        # ax1.scatter([x[0,0]], [y[0,0]], [z[0,0]], color='r', s=4, marker='o')
#        # ax1.scatter([x[0,1]], [y[0,1]], [z[0,1]], color='g', s=4, marker='o')
#        for i in range(nodes): #we draw the figure patch by patch
#            S=[]
#            if(i==nodes-1):
#                iPlusOne=0
#            else:
#                iPlusOne=i+1
#            for j in range(nodes):
#                if (j == nodes - 1):
#                    jPlusOne = 0
#                else:
#                    jPlusOne = j + 1
#                X = [[x[i, j], x[i, jPlusOne]],[x[iPlusOne, j],x[iPlusOne, jPlusOne]]]
#                Y = [[y[i, j], y[i, jPlusOne]], [y[iPlusOne, j], y[iPlusOne, jPlusOne]]]
#                Z = [[z[i, j], z[i, jPlusOne]], [z[iPlusOne, j], z[iPlusOne, jPlusOne]]]
#                colorpatch=self.getColorPatch(['red','blue','green'])
#                if(colorpatch=='red'):
#                    colorstate=0
#                if (colorpatch == 'blue'):
#                    colorstate = 1
#                if (colorpatch == 'green'):
#                    colorstate = 2
#                s = states(index=[i,j],color=colorstate,position=[x[i, j],y[i, j],z[i, j]], actions=self.setAction(i,[0,1,2,3],BiasIndex))
#                S.append(s)
#                if(i!=0):
#                    s.setLeftneighbor(self.environment.states[i-1][j])
#                    self.environment.states[i-1][j].setRightneighbor(s)
#                if(j!=0):
#                    s.setDownneighbor(S[j-1])
#                    S[j-1].setUpneighbor(s)
#                if(j==nodes-1):
#                    S[0].setDownneighbor(s)
#                    s.setUpneighbor(S[0])
#                if(i==nodes-1):
#                    s.setRightneighbor(self.environment.states[0][j])
#                    self.environment.states[0][j].setLeftneighbor(s)
#                ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, color=colorpatch, alpha=1, shade=True,linewidth=0)
#            self.environment.states.append(S)
#        self.explore(SizeOfTraining,BiasIndex,nodes)
#        # ax1.plot_wireframe(x, y, z, rstride=SpaceColumn, cstride=1, color='tan', alpha=0.7)
#        for l in range(self.PathTrain[0].__len__()):
#            ax1.plot_wireframe(self.PathTrain[0][l], self.PathTrain[1][l], self.PathTrain[2][l], rstride=SpaceColumn, cstride=1, color='tan', alpha=0.6, linewidth=4)
#        self.CalculateObservable(self.environment.Trajectory, TestData, stepSize, numIteration)
#        plt.show()
#
#
#
##function for robot exploring the environment
#    def explore(self,TrajectoryLentgh,BiasIndex,nodes):
#        Actions=[]
#        for i in range(TrajectoryLentgh):
#            List=[]
#            x=[]
#            y=[]
#            z=[]
#            (s1,s2)=self.randomStart(BiasIndex,nodes)
#            S = self.environment.states[s1][s2]
#            for j in range(5):
#                a=self.getAction(S)
#                Actions.append(a)
#                S=S.transitionToNextState(a)
#                observation=S.observation(self.environment, self.ErrorInObservation)
#                List.append((a,observation))
#                x.append(S.position[0])
#                y.append(S.position[1])
#                z.append(S.position[2])
#            self.PathTrain[0].append(x)
#            self.PathTrain[1].append(y)
#            self.PathTrain[2].append(z)
#            self.environment.Trajectory.append(List)
#        Up=0
#        Down=0
#        Right=0
#        Left=0
#        for i in Actions:
#            if(i==0):
#                Left+=1
#            elif(i==1):
#                Right+=1
#            elif(i==2):
#                Up+=1
#            elif(i==3):
#                Down+=1
#        print("Left Move:", Left/len(Actions))
#        print("Right Move:", Right/len(Actions))
#        print("Up Move:", Up / len(Actions))
#        print("Down Move:", Down/len(Actions))
#
## Construct matrices and calculate probabilities and produce results (Main Function)
#    def CalculateObservable(self,TrainingSet,TestData, stepSize, numIteration):
#        TestSet=TestData
#        RANK = self.environment.rank
#        O = self.constructDynamicMatriices(TrainingSet)
#        O_Modified = O[0]
#        HenkelMatrix = O[0][:, 0:37]
#        U, S, R = np.linalg.svd(HenkelMatrix)
#        print("Singular Values: ", S)
#        R = R[0:RANK, :]
#        b1 = HenkelMatrix[0, :] * R.T
#        P_THR = np.linalg.pinv(HenkelMatrix * R.T)
#        b_inf = P_THR * (HenkelMatrix[0, :].T)
#        Result1 = []
#        RealProbabilityResult = []
#        for i in TestSet:
#            r1 = self.RealProbabilityTest(i, state=A)
#            Result1.append('{:.10f}'.format(r1))
#            # if(r1==0):
#            #     r1=0.000000000000000001
#            RealProbabilityResult.append(r1)
#        # print(RealProbabilityResult)
#        Result2 = []
#        ResultWithoutDenoisng = []
#        j = 0
#        for i in TestSet:
#            r1 = self.FakeprobabilityTesWithOutDenoising(i, O, P_THR, R, b_inf, b1)[0, 0]
#            if (r1 < 0):
#                r1 = 0
#            ResultWithoutDenoisng.append('{:.10f}'.format(r1))
#            Result2.append(abs(r1 - RealProbabilityResult[j]))
#            j += 1
#        ListOfEqualEntities = self.FindEqualElementsOfDynamicMatrix(self.ExtendedTest, self.HISTORYelement)
#        # print(ListOfEqualEntities)
#        W = newAlgorithm(rank=RANK, DynamicMatrix=O_Modified, weights=self.environment.ExtendedWeight,
#                         ListEqualEntitities=ListOfEqualEntities, lambda1=self.lambda1, step=stepSize,
#                         numberOfStep=numIteration,
#                         FileName=self.FileName)  # the hyperparameter of second constarints are adjusted here
#        Q, R = W.GradientEngine()
#        if (type(Q) == bool):
#            if (Q == False):
#                self.R1, self.R2 = False, R
#                return
#        InverseR = R.T
#        InverseR = InverseR[:37, :]
#        NewP_TH = Q * R[:, :37]
#        Newb1 = (NewP_TH[0]) * InverseR
#        NewP_THR = np.linalg.pinv(NewP_TH * InverseR)
#        Newb_inf = NewP_THR * (NewP_TH[0]).T
#        j = 0
#        Result3 = []
#        ResultDenoising = []
#        for i in TestSet:
#            r1 = self.FakeprobabilityTesWithDenoising(i, Q, R, NewP_THR, InverseR, Newb_inf, Newb1)[0, 0]
#            if (r1 < 0):
#                r1 = 0
#            ResultDenoising.append('{:.10f}'.format(r1))
#            Result3.append(abs(r1 - RealProbabilityResult[j]))
#            j += 1
#        print("Result Without Denoising: ")
#        print(ResultWithoutDenoisng)
#        print("Result With Denoising: ")
#        print(ResultDenoising)
#        print("Real Probability :")
#        print(Result1)
#        REALHenkel = self.RealHenkelMatrix()  # Calculates the exact Henkel Matrix
#        ErrorHenkelWithoutDenoing = np.linalg.norm(REALHenkel - O[0][:, :37], 'fro')
#        ErrorHenkelWithDenoing = np.linalg.norm(REALHenkel - NewP_TH, 'fro')
#        AverageMethodWithoutDenoising = sum(Result2) / float(len(Result2))
#        AverageMethodWithDenoising = sum(Result3) / float(len(Result3))
#        self.R1, self.R2, self.E1, self.E2 = AverageMethodWithoutDenoising, AverageMethodWithDenoising, ErrorHenkelWithoutDenoing, ErrorHenkelWithDenoing
#
#        return
#
## computes and returns an array containing all the observable matrices inculiding P(H,T),B(H,ao,T) for all ao
#    def constructDynamicMatriices(self, TrainingSet):
#        H=self.combination()
#        T=self.combination()
#        self.TESTelemet = H
#        self.HISTORYelement = H
#        T = self.combinationExtended()
#        self.ExtendedTest = T
#        self.environment.Weight = np.zeros((H.__len__(), self.ExtendedTest.__len__()))
#        O1 = []
#        for history in self.HISTORYelement:
#            l = []
#            for test in self.ExtendedTest:
#                Output = self.ProbabilityTestGivenHistory(test, history, TrainingSet)
#                l.append(Output[1])
#                self.environment.Weight[self.environment.indexMap[self.unpack(history)], self.environment.indexMap[self.unpack(test)]] = \
#                Output[0]
#            O1.append(l)
#        self.environment.ExtendedWeight = np.matrix(self.environment.Weight)
#        O2 = {}
#        O1 = np.matrix(O1)
#        for action in self.environment.actions:
#            for observation in self.environment.observations:
#                O3 = []
#                l = np.matrix(O1[:, self.environment.indexMap[self.unpack([(action, observation)])]])
#                for test in self.TESTelemet:
#                    if (test.__len__() != 0):
#                        l = np.hstack((l, O1[:, self.environment.indexMap[self.unpack([(action, observation)] + test)]]))
#                O2[(action, observation)] = l
#        R = []
#        R.append(O1)
#        R.append(O2)
#        return R
#
#
## calculates the joint probability of a test given its history and its corresponding weights P(A_1,Y_1,A_2,Y_2, ... ,Y_n) = P(Y_1|A_1)*P(Y_2|A_2,A_1,Y_1) *** it is implemented via dynamic programming for faster computational time ***
#    def ProbabilityTestGivenHistory(self, Test, History, TrainingSet):
#        TEST = History + Test
#        if self.unpack(TEST) in self.environment.PTH:
#            # print('Failed')
#            return (self.environment.PTH[self.unpack(TEST)][0],self.environment.PTH[self.unpack(TEST)][1])
#        if (TEST.__len__() == 0):
#            self.environment.PTH[self.unpack(TEST)] = [1,1]
#            return (1,1)
#        ProbabilityOfTest=1
#        for i in range(TEST.__len__()):
#            p=self.ComplementaryProbabilty(TEST[i],TEST[:i],TrainingSet)
#            ProbabilityOfTest=ProbabilityOfTest*p
#        HistoryCounter = 0
#        for o in TrainingSet:
#            Test1=self.unpack(Test[:len(Test)-1])+str(Test[len(Test)-1][0])
#            if self.TwoGivenListIsEqual(Test1, self.unpack(o)):
#                HistoryCounter += 1
#        if (HistoryCounter == 0):
#            self.environment.PTH[self.unpack(TEST)] = [0,0]
#            return (0,0)
#        weight = HistoryCounter / len(TrainingSet)
#        self.environment.PTH[self.unpack(TEST)] = [weight,ProbabilityOfTest]
#        return (weight, ProbabilityOfTest)
#
#
## This is a complementray function for def ProbabilityTestGivenHistory. It calculates P(Y_i|Y_1,A_1,Y_2, ..., A_i-1)
#    def ComplementaryProbabilty(self,y,history,TrainingSet):
#        Test=history+[y]
#        if (self.unpack(Test) in self.environment.ConditionalPTH):
#            return self.environment.ConditionalPTH[self.unpack(Test)]
#        else:
#            HistoryCounter = 0
#            HistoryDenumenatorCounter = 0
#            Test=self.unpack(history)
#            Test=Test+self.unpack(y)
#            Test2=self.unpack(history)+str(y[0])
#            for data in TrainingSet:
#                if(self.TwoGivenListIsEqual(Test,self.unpack(data))):
#                    HistoryCounter+=1
#                if(self.TwoGivenListIsEqual(Test2,data)):
#                    HistoryDenumenatorCounter+=1
#            if (HistoryDenumenatorCounter == 0):
#                self.environment.ConditionalPTH[self.unpack(Test)]=0
#                return 0
#            else:
#                self.environment.ConditionalPTH[self.unpack(Test)] = HistoryCounter/HistoryDenumenatorCounter
#                return HistoryCounter/HistoryDenumenatorCounter
#
#    # construct the list for second constraints in denoising algorithm
#    def FindEqualElementsOfDynamicMatrix(self, Test, History):
#        MainList = []  # MainList consist of all combination Test+History where in each of its element first entity is the combination, second entity is the index of corresponding element in dynamic matrix
#        for test in Test:
#            for history in History:
#                MainList.append([history + test, [self.environment.indexMap[self.unpack(history)], self.environment.indexMap[self.unpack(test)]]])
#        ListOfEqualEntities = []
#        while len(MainList) != 0:
#            Flag, Q, MainList = self.FindEqualities(MainList[0], MainList)
#            if (Flag):
#                ListOfEqualEntities.append(Q)
#        return ListOfEqualEntities
#
#    # supporting function for FindEqualElementsOfDynamicMatrix
#    def FindEqualities(self, i, List):
#        Q = []
#        Q.append(i[1])
#        Flag = False
#        List.remove(i)
#        for j in List:
#            if (j[0] == i[0]):
#                Q.append(j[1])
#        if len(Q) > 1:
#            Flag = True
#        for k in range(len(Q)):
#            if (k != 0):
#                List.remove([i[0], Q[k]])
#        return Flag, Q, List
#
#
#
#
#    #Assigns color randomley to the patches based on the number of observation List is the list of colors
#    def getColorPatch(self,List):
#        r=random.uniform(0,1)
#        l=len(List)
#        k=1/l
#        step=0
#        for i in range(l):
#            if(step<=r<k+step):
#                return List[i]
#            else:
#                step+=k
#        return List[l-1]
#
## checks whether based on the position of the given point it can turn just left, right or it can also goes up and downs.
#    def setAction(self,i,Actions,BiasIndex):
#        if(i%BiasIndex!=0):
#            return [1/2,1/2,0,0]
#        else:
#            return [1/4,1/4,1/4,1/4]
#
##given state return the action of trajectory based on the possible valid action for that state
#    def getAction(self,state: states):
#        step=0
#        r=random.uniform(0,1)
#        for l in range(len(state.actions)):
#            if(step<=r<step+state.actions[l]):
#                return l
#            else:
#                step+=state.actions[l]
#        return len(state.actions)-1
#
## Assigns the starting point of robot randomely
#    def randomStart(self,BiasIndex,nodes):
#        L=[]
#        for n in range(nodes):
#            if(n%BiasIndex==3):
#                L.append(n)
#        r=random.uniform(0,1)
#        p=1/nodes
#        step=0
#        j1=0
#        for i in range(nodes):
#            if(step<=r<p+step):
#                j1=i
#                break
#            else:
#                step+=p
#        i1=L[len(L)-1]
#        step=0
#        p=1/len(L)
#        r = random.uniform(0, 1)
#        for j in L:
#            if (step <= r < p + step):
#                i1=j
#                break
#            else:
#                step+=p
#        return (i1,j1)
#
#
#    def unpack(self, list):
#        s = ''
#        for l in list:
#            s = s + str(l[0])
#            for k in l[1]:
#                s=s+str(k)
#        return s
#
#
#    def TwoGivenListIsEqual(self, L1, L2):
#        for i in range(L1.__len__()):
#            if (L1[i] != L2[i]):
#                return False
#        return True
#
#
#    # create all the permutation of 0,1,2 of given size i
#    def combination(self):
#        T1 = [[]]
#        T2 = []
#        for i1 in [0, 1,2,3]:
#            for i2 in [0,1,2]:
#                for i3 in [0,1,2]:
#                    T1.append([(i1, [i2,i3])])
#                    #Adding more states makes it too big
#                    # for i4 in [0, 1,2,3]:
#                    #     for i5 in [0, 1, 2]:
#                    #         for i6 in [0,1,2]:
#                    #             T2.append([(i1, [i2,i3]),(i4, [i5,i6])])
#        return T1
#
#    def combinationExtended(self):
#        T1 = [[]]
#        T2 = []
#        for i1 in [0, 1,2,3]:
#            for i2 in [0,1,2]:
#                for i3 in [0,1,2]:
#                    T1.append([(i1, [i2,i3])])
#                    for i4 in [0, 1,2,3]:
#                        for i5 in [0, 1, 2]:
#                            for i6 in [0,1,2]:
#                                T2.append([(i1, [i2,i3]),(i4, [i5,i6])])
#        List=T1+T2
#        for element in List:
#            self.environment.indexMap[self.unpack(element)]=List.index(element)
#        return T1+T2
#
