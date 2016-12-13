#we represent all actions, states and observations as numbers. e.g. 0,1,2,3,... corresponds to action a,b,c,d,....
# In a similar manner 0,1,2,3,... corresponds to states A,B,C,D,.... In a similar way 0,1,2,3,... coreesponds to Observation 0,1,2,3,....
#probabilityTrans is a list where its ith component corresponds to transition probobility to the next states by taking action i
#observationProbability is a list where its ith component corresponds to probobility of observing i

import sys
import random

class Hmmstate:

    actions=[]
    def __init__(self,probabilityTrans,observationProbability,Actions,index):
        self.observationProbability=observationProbability
        self.transition=probabilityTrans
        self.actions=Actions
        self.index=index

#For given a state and action produces the next state
    def transToNextState(self,action,states):
        ProbableNextStates=self.transition[action]
        s=0
        for state in ProbableNextStates:
            s+=state
        if(s==0):
            print("It is not working good!")
            sys.exit("aa! errors!")
        i= random.uniform(0,1)
        FirstLine = 0
        for j in range(len(ProbableNextStates)):
            if (i >= FirstLine and i < (FirstLine + ProbableNextStates[j])):
                return states[j]
            else:
                FirstLine += ProbableNextStates[j]
        return  states[len(states)-1]   #for making sure that we are not going to encounter 'None'
