import random

class environment:

    # 0 stands for red observation, 1 stands for blue observation, 2 stands for green observation
    def __init__(self, possibleActions, possibleObservation,rank):
        self.StartingState=[]
        self.rank = rank
        self.states=[]
        self.HmmStates=[]
        self.actions=possibleActions
        self.observations=possibleObservation
        self.Trajectory=[] #Contains trajectory for the experiment
        self.TrainingSet=[]
        self.PTH={} # PTH is a dictionary of all joint probabilty of test and histories for without Denoing  ***used for dynamic programming***
        self.ConditionalPTH={} # the same as PTH but instead restore conditional probability of P(Y_i|Y_1,...)
        self.indexMap = {}  # used for finiding the index of a given element inside a list(in B_TaoH is used)
        self.DenoisingPTH={} # DenoinsingPTH is a dictionary of all joint probabilty of test and histories for Denoing Method
        self.ExtendedWeight = []

# For given a state produces the corresponding observation
    def observe(self,state,size):
        probableObservation=state.observationProbability
        List=[]
        for i in range(size):
            List.append(-1)
        for i in range(size):
            r = random.uniform(0, 1)
            FirstLine = 0
            for j in range(len(probableObservation)):
                if (r >= FirstLine and r < (FirstLine + probableObservation[j])):
                    List[i]=j
                    break
                else:
                    FirstLine += probableObservation[j]
            if(List[i]==-1):
                List[i]=len(probableObservation)-1
        return List[0]

# Given a list [a,b] for tuple observation, returns its corresponding number.
    def ConvertColor(self,List):
        return List[0]+3*List[1]

