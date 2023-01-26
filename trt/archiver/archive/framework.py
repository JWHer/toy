from archive.common import ArchiveDict, ArchiveObject

import abc
import configparser
from os import chdir, getcwd
from os.path import basename, exists
from shutil import copyfile
from urllib.parse import urlparse

class Framework(ArchiveDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, type=str(), **kwargs):
        words = type.split('.')
        if 'deepstream' == words[1]:
                return DeepStreamPlugin.create(type='.'.join(words[2:]), **kwargs)


class DeepStreamPlugin(ArchiveDict):
    def __init__(self, type=None, **kwargs):
        super().__init__(**kwargs)
        self._type = type if type else None
        self._links = list()
        self._properties = dict()
        self._attributes = dict()
        self._hints = dict()

    @property
    def type(self):
        return self._type

    @property
    def links(self):
        return self._links

    @property
    def properties(self):
        return self._properties

    @property
    def attributes(self):
        return self._attributes

    @property
    def hints(self):
        return self._hints

    @links.setter
    def links(self, value):
        self._links = value

    @properties.setter
    def properties(self, value):
        self._properties = value

    @attributes.setter
    def attributes(self, value):
        self._attributes = value

    @hints.setter
    def hints(self, value):
        self._hints = value

    @abc.abstractmethod
    def _visit_to_compose(self):
        dumped = ArchiveObject.dump(self)
        for key, value in dumped.items():
            if key == 'links':
                self._links = value
            if key == 'properties':
                self._properties = value
            if key == 'attributes':
                self._attributes = value
            if key == 'hints':
                self._hints = value

    @classmethod
    def create(cls, type=str(), **kwargs):
        if type == 'nvinfer':
            return NvInfer(**kwargs)

class NvInfer(DeepStreamPlugin):
    def __init__(self, **kwargs):
        super().__init__(type='nvinfer', **kwargs)
        self._config = configparser.ConfigParser()
        self._config['property'] = {}
        self._config['property']['gpu-id'] = '0'
        self._config['property']['net-scale-factor'] = '0.0039215697906911373'
        self._config['property']['model-color-format'] = '0'
        self._config['property']['num-detected-classes'] = '1'
        self._config['property']['interval'] = '0'
        self._config['property']['gie-unique-id'] = '1'
        self._config['property']['process-mode'] = '1'
        self._config['property']['network-type'] = '0'
        self._config['property']['cluster-mode'] = '4'
        self._config['property']['maintain-aspect-ratio'] = '1'
        self._config['property']['network-mode'] = '2'
        self._config['property']['batch-size'] = '1'
        self._config['property']['offsets'] = '0.0;0.0;0.0'

    @abc.abstractmethod
    def _visit_to_compose(self):
        super()._visit_to_compose()

        dumped = ArchiveObject.dump(self)
        if 'configs' in dumped.keys():
            for section, item in dumped['configs'].items():
                self._config[section] = item
        self.properties['config-file-path'] = '_'.join([self.name, self.type, 'config.txt'])

    @abc.abstractmethod
    def _visit_to_package(self):
        files = []
        for k, v in self._config['property'].items():
            if k not in ['custom-lib-path', 'model-engine-file', 'labelfile-path']:
                continue
            if type(v) != str:
                continue

            parsed = urlparse(v)
            if parsed.scheme != 'file' or exists(parsed.path) == False:
                continue

            self._config['property'][k] = filename = '/'.join(['.', basename(v)])
            copyfile(parsed.path, filename)
            filename = '/'.join(['.', basename(v)])
            files.append(filename)

        # ship a config file of the nvinfer
        filename = self.properties['config-file-path']
        with open(filename, 'w') as file:
            self._config.write(file)
            self.properties['config-file-path'] = filename = '/'.join(['.', filename])
            files.append(filename)
        return files

if __name__ != '__main__':
    ArchiveObject.register_creator('@framework.deepstream', Framework.create)