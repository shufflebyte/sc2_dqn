class BaseManager:
    def __init__(self):
        pass

    @property
    def get_name(self):
        return self.__class__.__name__
