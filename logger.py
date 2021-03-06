from datetime import datetime

class LogColors:
    INPUT = '\033[95m'
    INFO = '\033[94m'
    DEBUG = '\033[96m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Logger:

    @staticmethod
    def status(*args):
        print(LogColors.YELLOW + "[STATUS]" + LogColors.END + " -", *args)

    @staticmethod
    def debug(*args):
        print(LogColors.DEBUG + "[DEBUG]" + LogColors.END + " -", *args)

    @staticmethod
    def info(*args):
        print(LogColors.INFO + "[INFO]" + LogColors.END + " -", *args)

    @staticmethod
    def input(*args):
        print(LogColors.INPUT + "[INPUT]" + LogColors.END + " -", *args)

    @staticmethod
    def error(*args):
        print(LogColors.FAIL + "[ERROR]" + LogColors.END + " " + datetime.now().strftime("%H:%M:%S") + " -", *args)
