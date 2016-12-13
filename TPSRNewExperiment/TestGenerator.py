#Creates test data for our algorithm
from environment import *
import copy

class TestGenerator:

    def __init__(self,sizeaTest,environment):
        self.env=environment
        self.size=sizeaTest

    def generate(self):
        ListTest=[] #contains sublist of size 2,4,.. eventually this function returns ListTest = [[0,0],[0,1],...,[0,0,0,0],[1,0,0,0],...]
        for i in range(self.size):
            Temp=[]
            ListTest2 = copy.deepcopy(ListTest)
            ListTest3= copy.deepcopy(ListTest)
            for action in self.env.actions:
                for observation in self.env.observations:
                    Temp.append([action,observation])
            ListTest=copy.deepcopy(Temp)
            for element in Temp:
                for i in range(len(ListTest3)):
                    ListTest2[i] = element + ListTest2[i]
                ListTest=ListTest+ListTest2
                ListTest2=copy.deepcopy(ListTest3)
        return ListTest