import os
import signal
import os.path as osp

from datetime import datetime


class Logger:
    def __init__(self, log_dir=None):
        self.format = '%Y-%m-%d %H:%M:%S: '
        self.file = None
        self.log_dir = log_dir

        if log_dir is not None:
            path = osp.join(log_dir, 'log.txt')
            if not osp.exists(log_dir):
                os.mkdir(log_dir)
            self.file = open(path, 'a')
    
    def __get_prefix(self):
        return datetime.now().strftime(self.format)

    def log(self, message, no_prefix=False):
        """
        print message in terminal and file(if log_dir is given)
        """
        if not no_prefix:
            message = self.__get_prefix() + message
        print(message)
        if self.file is not None:
            print(message, file=self.file, flush=True)

    def set_interupt_message(self, message):
        """
        set output information after Ctrl-C interupt.
        """
        def handler(signum, frame):
            nonlocal message
            self.log(message)
            exit()
        signal.signal(signal.SIGINT, handler)
