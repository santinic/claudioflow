
class SharedMemory:
    def __init__(self, initialize):
        self.mem = initialize

    def set(self, value):
        self.mem = value

    def get(self):
        return self.mem
