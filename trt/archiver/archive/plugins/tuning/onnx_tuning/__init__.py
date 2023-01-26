import sys


__all__ = ['ima_plugin_tuning_initialize']


def ima_plugin_tuning_initialize(**kwargs):
    return TuningPlugin(**kwargs)


class TuningPlugin:
    def __init__(self, **kwargs):
        pass
        
    def process(self, source):
        pass