import os

ROOT = os.getcwd()

class JSONConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])
