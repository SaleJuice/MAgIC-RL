import numpy as np


class Agent(object):
    '''Base class for all rl agents.
    '''
    
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        raise NotImplementedError

    def set_train(self) -> None:
        raise NotImplementedError
    
    def set_eval(self) -> None:
        raise NotImplementedError

    def save_model(self) -> None:
        raise NotImplementedError
    
    def load_model(self) -> None:
        raise NotImplementedError
    
    def get_action(self) -> np.ndarray:
        raise NotImplementedError

    def update(self) -> dict:
        raise NotImplementedError
