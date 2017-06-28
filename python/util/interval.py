import average
import monotonic

class IntervalTracker:
    def __init__(self, samplesize):
        self.accumulator = average.AverageAccumulator(samplesize)
        self.last_event = None

    def report_event(self):
        now = monotonic.monotonic()
        if self.last_event is None:
            self.last_event = now
        else:
            self.accumulator.append(now - self.last_event)
            self.last_event = now

    def estimate_interval_secs(self):
        return self.accumulator.get_average()

if __name__ == '__main__':
    tracker = IntervalTracker(10)
    for i in range(20):
        tracker.report_event()
        print(tracker.estimate_interval_secs())
