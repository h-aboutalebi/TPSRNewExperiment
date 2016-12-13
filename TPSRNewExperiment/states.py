import random
from environment import *
# we define the states of the environment here

class states:

    #The color associated with the state is the color of patch on the southeast of its position
    # 0 stands for red observation, 1 stands for blue observation, 2 stands for green observation
    #index refers to the location of state in the 2D array attribute 'environment.states' in the class environment
    def __init__(self,index,color,position,actions): # index is the index of the states
        self.color=color # the color associated with this state it is the color of patch on the Northeast of its position
        self.position=position # coordinates of 3D space related to this state
        self.actions=actions #actions can be performed in this state [0,1,2,3] corresponding to [left,right,up, down]
        self.Leftneighbor=None
        self.Rightneighbor=None
        self.Upneighbor=None
        self.Downneighbor = None
        self.index=index

# assigns the left neighbor
    def setLeftneighbor(self,state):
        self.Leftneighbor=state

# assigns the right neighbor
    def setRightneighbor(self,state):
        self.Rightneighbor=state

# assigns the up neighbor
    def setUpneighbor(self,state):
        self.Upneighbor=state

# assigns the down neighbor
    def setDownneighbor(self,state):
        self.Downneighbor=state

#given an action specifies the next state for transitioning
    def transitionToNextState(self, action):
        if (action==0):
            return self.Leftneighbor
        elif(action==1):
            return self.Rightneighbor
        elif(action==2):
            return self.Upneighbor
        elif(action==3):
            return self.Downneighbor

#observation function returns the observation associated with this state.
#p is the probability in [0,1] of possible error in perception of robot
    def observation(self,environment : environment,p : int):
        NE=self.color
        SE=self.Downneighbor.color
        # NW=self.Leftneighbor.color
        # SW=self.Leftneighbor.Downneighbor.color
        r=random.uniform(0,1)
        TrueColor=[NE,SE]
        WrongColor=self.RandomObservation(environment.observations,2)
        if (0<=r<p):
            return self.ConvertColor(WrongColor)
        else:
            return self.ConvertColor(TrueColor)


    def RandomObservation(self,Colors,size):
        p=1/len(Colors)
        S=[]
        for i in range(size):
            step=0
            r=random.uniform(0,1)
            Flag=True
            for j in Colors:
                if(step<=r<p+step):
                    S.append(j)
                    Flag=False
                    break
                else:
                    step+=p
            if(Flag):
                S.append(Colors[len(Colors)-1])
        return S

    # Given a list [a,b] for tuple observation, returns its corresponding number.
    def ConvertColor(self,List):
        return List[0]+3*List[1]
