import states
import state
import random
from cnn import Cnn
class Agent(object):

    def __init__(self, learning=True, epsilon = 1, alpha=0.1, discount=1, tolerance=0.01):

        self.Q = dict()
        self.Q1 = dict()
        self.Q2 = dict()

        self.s = states.States()
        self.s.buildAllStates()

        self.cnn = Cnn()

        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.tolerance = tolerance
        self.t = 2

        self.convolutionLayersNumber = 0
        self.poolingLayersNumber = 0
        self.fullyConnectedLayersNumber = 0

        self.convolutionLayersLimit = 3
        self.poolingLayersLimit = 2
        self.fullyConnectedLayersLimit = 2

        self.actionsInitialState()
        self.actionsConvolutionState()
        self.actionsPoolingState()
        self.actionsFullyConnectedState()

    def reset(self, testing = False):

        if testing == True:
            self.epsilon = 0
            self.alpha = 0
        else:
            self.convolutionLayersNumber = 0
            self.poolingLayersNumber = 0
            self.fullyConnectedLayersNumber = 0
            self.epsilon = 0.9999 * self.t
            del self.cnn
            self.cnn = Cnn()

    def actionsInitialState(self):

        actions = dict()
        self.initialActions = []
        statesHere = self.s.conv + self.s.pool + self.s.fully + self.s.terminate
        for eachState in statesHere:
            self.initialActions.append(eachState)
            actions[eachState] = 0.0

        self.initial = state.InitialState()

        self.Q[self.initial] = self.Q1[self.initial] = \
        self.Q2[self.initial] = actions


    def actionsConvolutionState(self):

        actions = dict()
        self.convolutionActions = []
        statesHere = self.s.conv + self.s.pool + self.s.fully + self.s.terminate
        for eachState in statesHere:
            self.convolutionActions.append(eachState)
            actions[eachState] = 0.0

        statesHere = self.s.conv
        for eachState in statesHere:
            self.Q[eachState] = self.Q1[eachState] = \
                self.Q2[eachState] = actions

    def actionsPoolingState(self):

        actions = dict()
        self.poolingActions = []
        statesHere = self.s.conv + self.s.fully + self.s.terminate
        for eachState in statesHere:
            self.poolingActions.append(eachState)
            actions[eachState] = 0.0

        statesHere = self.s.pool

        for eachState in statesHere:
            self.Q[eachState] = self.Q1[eachState] = \
                self.Q2[eachState] = actions

    def actionsFullyConnectedState(self):

        actions = dict()
        self.fullyConnectedActions = []
        statesHere = self.s.fully + self.s.terminate
        for eachState in statesHere:
            self.fullyConnectedActions.append(eachState)
            actions[eachState] = 0.0

        statesHere = self.s.fully

        for eachState in statesHere:
            self.Q[eachState] = self.Q1[eachState] = \
                self.Q2[eachState] = actions

    def getMaxQValue(self,QTable,state):
        #print('Debug::States for maxQ:',state)
        if state.name == states.TERMINATE:
            return 0.0
        maxQ = max(QTable[state].items(), key=lambda x: x[1])[1]

        return maxQ

    def getMaxQState(self,QTable,state):
        maxState = max(QTable[state].items(), key=lambda x: x[1])[0]
        return maxState

    def update_Q1_table(self, stateActionPair, reward):
        #print('StateActionPair',stateActionPair)
        #print('reward Here',reward)
        for eachStateAction in stateActionPair:
            print('Update for :',eachStateAction)
            self.learn(eachStateAction[0], eachStateAction[1], reward)

    def learn(self, state, action, reward):
        print('State', state)
        print('Action', action)
        print('Q-value added', self.Q[state][action])

        if self.learning and action.name!=states.TERMINATE:
            getMaxQValueOfNextState = self.getMaxQValue(self.Q,\
                                                        self.getMaxQState(self.Q,action))
            #print('getMaxQValueOfNextState',getMaxQValueOfNextState)
            #print('reward',reward)
            self.Q[state][action] = ((1-self.alpha) * self.Q[state][action]) + \
                                (self.alpha*(reward +(self.discount * getMaxQValueOfNextState)))
        else:
            self.Q[state][action] = ((1 - self.alpha) * self.Q[state][action]) + \
                                    (self.alpha * reward)

        print('at learn',self.Q)

        print('State',state)
        print('Action',action)
        print('Q-value added',self.Q[state][action])
        print('=====================================================================')

    def chooseAction(self,actionItems,state):
        if self.learning:
            if self.epsilon>random.random():
                action = random.choice(actionItems)
            else:
                action = max(self.Q[state].items(), key = lambda x: x[1])[0]

        return action

    def setInitialState(self,changeState):
        if changeState.name == states.CONV:
            self.convolutionLayersNumber += 1

        elif changeState.name == states.POOL:
            self.poolingLayersNumber += 1

        elif changeState.name == states.FULLY:
            self.fullyConnectedLayersNumber +=1

    def update(self):
        print('New Code')
        while self.epsilon > self.tolerance:
            #Begin trial
            convNet = ()
            stateAction = ()
            changeState = self.initial
            actionTaken = self.chooseAction(self.initialActions, changeState)
            stateAction += ((changeState, actionTaken),)
            changeState = actionTaken
            convNet += (changeState,)
            self.setInitialState(changeState)
            #print('Initial Change state to be added', changeState)

            while changeState.name != states.TERMINATE:
                if changeState.name == states.CONV and \
                                self.convolutionLayersNumber<self.convolutionLayersLimit:
                    #print('states.CONV getting called')
                    actionTaken = self.chooseAction(self.convolutionActions, changeState)
                    stateAction += ((changeState, actionTaken),)
                    changeState = actionTaken
                    self.convolutionLayersNumber += 1

                elif changeState.name == states.POOL and \
                                self.poolingLayersNumber<self.poolingLayersLimit:
                    #print('states.POOL getting called')
                    actionTaken = self.chooseAction(self.poolingActions, changeState)
                    stateAction += ((changeState, actionTaken),)
                    changeState = actionTaken
                    self.poolingLayersNumber += 1

                elif changeState.name == states.FULLY and \
                                self.fullyConnectedLayersNumber<self.fullyConnectedLayersLimit:
                    #print('states.FULLY getting called')
                    actionTaken = self.chooseAction(self.fullyConnectedActions, changeState)
                    stateAction += ((changeState, actionTaken),)
                    changeState = actionTaken
                    self.fullyConnectedLayersNumber += 1
                    print('FullyConnectedLayersNumber--->',self.fullyConnectedLayersNumber)

                else:
                    #print('Else getting called in agent')
                    actionTaken = self.s.terminate[0]
                    stateAction += ((changeState, actionTaken),)
                    changeState = actionTaken
                print('Change state to be added',changeState)
                convNet += (changeState,)



            print('Convnet--->>>>',convNet)
            score = self.cnn.buildModel(convNet)
            print(score)
            self.update_Q1_table(stateAction, score[1])
            #print('Model Trained!!')
            print('=======self.Q======',self.Q)
            self.reset()

    def test(self):
        print('---------------Start Testing-----------')
        convNet = ()
        self.cnn = Cnn()
        changeState = self.initial
        convNet += (changeState,)
        while changeState.name != states.TERMINATE:

            changeState = self.getMaxQState(self.Q,changeState)
            convNet += (changeState,)

        print('TestingnConvnet',convNet)
        score = self.cnn.buildModel(convNet)
        print('TestScore',score)



if __name__=='__main__':
    a = Agent()
    a.update()
    a.test()

#4000000