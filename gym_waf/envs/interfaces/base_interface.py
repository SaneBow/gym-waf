class ClassificationFailure(Exception):
    pass


class LocalInterface(object):
    def __init__(self) -> None:
        pass
    
    def get_score(self, payload):
        raise NotImplementedError("get local score not implemented")

    def get_label(self, payload):
        raise NotImplementedError("get local label not implemented")


class RemoteInterface(object):
    def __init__(self) -> None:
        pass
    
    def get_score(self, payload):
        raise NotImplementedError("get remote score not implemented")

    def get_label(self, payload):
        raise NotImplementedError("get remote label not implemented")



