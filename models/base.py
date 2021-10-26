from abc import abstractmethod, abstractproperty


class BaseAlgorithm:
        

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def print(self):
        pass