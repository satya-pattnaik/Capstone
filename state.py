class State(object):

    def __init__(self,name=''):
        self.name = name

class ConvolutionState(State):

    def __init__(self, filterSize, numberOfFilters, name=''):
        self.filterSize = filterSize
        self.numberOfFilters = numberOfFilters
        super(ConvolutionState, self).__init__(name)

class PoolingState(State):

    def __init__(self, poolSize, strides, name=''):
        self.poolSize = poolSize
        self.strides = strides
        super(PoolingState, self).__init__(name)

class FullyConnectedState(State):

    def __init__(self, numberOfNeurons, name=''):
        super(FullyConnectedState, self).__init__(name)
        self.numberOfNeurons = numberOfNeurons

class TerminationState(State):

    def __init__(self, name='Terminate'):
        super(TerminationState, self).__init__(name)


class InitialState(State):

    def __init__(self, name='Initial'):
        super(InitialState, self).__init__(name)