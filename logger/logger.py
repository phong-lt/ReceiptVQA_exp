import sys

class Logger(object):
    def __init__(self, filename):
        self.filename = filename

    class Transcript:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        def __getattr__(self, attr):
            return getattr(self.terminal, attr)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass

    def start(self):
        sys.stdout = self.Transcript(self.filename)

    def stop(self):
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal