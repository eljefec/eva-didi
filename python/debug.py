import logging
import mem_top
import signal
import sys
import threading
import time
import traceback

class Debugger:
    def __init__(self, logname, freqsecs = 30 * 60):
        logging.basicConfig(filename = logname, level = logging.DEBUG)

        self.freqsecs = freqsecs

        signal.signal(signal.SIGUSR1, self.dump)
        # signal.signal(signal.SIGQUIT, self.dumpstacks)

        self.timer = None
        self.debug()

    def debug(self):
        self.dump(None, None)

        self.timer = threading.Timer(self.freqsecs, self.debug)
        self.timer.start()

    def shutdown(self):
        if self.timer:
            self.timer.cancel()

    def dump(self, signal, frame):
        if signal:
            logging.debug('Received signal: ' + str(signal))

        self.dumpmem()
        self.dumpstacks()

    def dumpmem(self):
        logging.debug(mem_top.mem_top())

    def dumpstacks(self):
        id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
        for threadId, stack in sys._current_frames().items():
            logging.debug("# Thread: %s(%d)" % (id2name.get(threadId,""), threadId))
            for filename, lineno, name, line in traceback.extract_stack(stack):
                logging.debug('File: "%s", line %d, in %s' % (filename, lineno, name))
                if line:
                    logging.debug("  %s" % (line.strip()))

if __name__ == '__main__':
    debugger = Debugger('logs/debuggertest.log', 1)
    while True:
        time.sleep(5)

    debugger.shutdown()
