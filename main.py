import os
import sys
from colorama import init
from predict import case_study
init(autoreset=True) 
print('当前的工作路径为： ' + os.getcwd()) # Display the current path
path = os.path.abspath('.')
sys.path.append(path)

class Logger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        if not self.log.closed:  # Check if the file is closed
            self.terminal.flush()
            self.log.flush()

    def close(self):
        if not self.log.closed:  # Ensure that the file is not closed multiple times
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

with Logger('Diary_GCNMKLSDA.txt', sys.stdout) as logger:
    sys.stdout = logger
    
    case_study()
    print('\n',"All results have been written to both console and file")