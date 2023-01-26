from archive.common import ArchivePlugin

import abc


__all__ = ['ima_plugin_model_initialize']


def ima_plugin_model_initialize(**kwargs):
    return ModelPlugin(**kwargs)


class ModelPlugin(ArchivePlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def _visit_to_compose(self):
        pass
    
    @abc.abstractmethod
    def _visit_to_package(self):
        pass