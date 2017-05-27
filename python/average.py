import collections

class AverageAccumulator:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.samples = collections.deque()
        self.sum = 0
        self.average = None

    def append(self, value):
        self.samples.append(value)
        self.sum += value

        if (len(self.samples) >= self.maxsize):
            popped = self.samples.popleft()
            self.sum -= popped

        self.average = self.sum / len(self.samples)

    def get_average(self):
        return self.average
