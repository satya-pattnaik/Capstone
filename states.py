import state

CONV = "Conv"
POOL = "Pool"
FULLY = "Fully"
TERMINATE = "Terminate"
INITIAL = "Initial"

class States(object):

    def __init__(self):

        self.conv = []
        self.pool = []
        self.fully = []
        self.terminate = []

        self.convolutionFilterSizes = [1, 3, 5]
        self.numberOfConvolutionFilter = [64, 128, 256]

        self.poolingHyperparameters = [(5,3), (3,2), (2,2)]

        self.fullyConnectedNeurons = [128, 256,512]





    def buildConvolutionStates(self):
        for i in self.convolutionFilterSizes:
            for j in self.numberOfConvolutionFilter:
                conv = state.ConvolutionState(i,j,CONV)
                self.conv.append(conv)


    def buildPoolingStates(self):
        for i in self.poolingHyperparameters:
            pool = state.PoolingState(i[0],i[1],POOL)
            self.pool.append(pool)

    def buildFullyConnectedStates(self):
        for i in self.fullyConnectedNeurons:
            fullyConnected = state.FullyConnectedState(i,FULLY)
            self.fully.append(fullyConnected)

    def buildTerminationStates(self):
        self.terminate.append(state.TerminationState(TERMINATE))

    def buildAllStates(self):
        self.buildConvolutionStates()
        self.buildPoolingStates()
        self.buildFullyConnectedStates()
        self.buildTerminationStates()


if __name__ == '__main__':
    states = States()
    states.buildAllStates()
