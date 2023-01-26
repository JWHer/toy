import abc
from collections import deque
import importlib
from os import chdir, getcwd, makedirs
from os.path import abspath, basename, exists
import shutil
import tarfile
import time
import zipfile


def create_plugin_package(type, uri, **kwargs):
    words = __name__.split('.')
    name = '.'.join(['.'.join(words[:-1]), 'plugins', type, uri])
    mod = importlib.import_module(name)
    return getattr(mod, '_'.join(['ima_plugin', type, 'initialize']))(**kwargs)


class ArchiveObject:
    factory = {}

    def __init__(self, parent=None, name=str(), value=None, data=None, manual_parse=False):
        self._parent = parent
        self._name = name if name else str()
        self._value = value if value else None
        self._data = data if data else None
        self._manual_parse = manual_parse
        if not self._manual_parse:
            ArchiveObject.parse_recursive(self)

    def __repr__(self):
        return "<{}:{} name={} data={}>".format(self.__class__.__name__, hex(id(self)), self._name, self._data)

    @property
    def parent(self):
        return self._parent

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @property
    def data(self):
        return self._data

    @property
    def ancestor(self):
        if self._parent:
            return self._parent.ancestor
        return self

    @property
    def address(self):
        if not self.parent:
            address = ''
        else:
            address = self.parent.address
            if address:
                address = '.'.join([self.parent.address, self.parent.get_id_of(self)])
            else:
                address = self.parent.get_id_of(self)
        return address

    @data.setter
    def data(self, value):
        self._data = value

    def get_id_of(self, obj):
        if type(self.data) is dict:
            for k, v in self.data.items():
                if v is obj:
                    return k
        elif type(self.data) is list:
            for i, v in enumerate(self.data):
                if v is obj:
                    return str(i)
        return ''

    def update(self, value):
        self._value = value
        self._data = None
        ArchiveObject.parse_recursive(self)

    def query_archive_object(self, address, from_ancestor=True):
        def _query_recursive_helper(_obj, _address):
            if not _address:
                return _obj

            words = deque(_address.split('.'))
            key = words.popleft()
            if type(_obj.data) is dict:
                if key in _obj.data.keys():
                    return _query_recursive_helper(_obj.data[key], '.'.join(words))
            elif type(_obj.data) is list:
                key = int(key)
                if 0 <= key < len(_obj.data):
                    return _query_recursive_helper(_obj.data[key], '.'.join(words))
            return None

        obj = self.ancestor if from_ancestor else self
        return _query_recursive_helper(obj, address)

    def visit_to_resolve_recursive(self):
        self._visit_to_resolve()
        if type(self.data) is dict:
            for k, v in self.data.items():
                v.visit_to_resolve_recursive()
        elif type(self.data) is list:
            for v in self.data:
                v.visit_to_resolve_recursive()

    def visit_to_compose_recursive(self):
        if type(self.data) is dict:
            for v in self.data.values():
                v.visit_to_compose_recursive()
        self._visit_to_compose()

    def visit_to_package_recursive(self):
        files = list()
        if type(self.data) is dict:
            for v in self.data.values():
                packaged = v.visit_to_package_recursive()
                if packaged:
                    files.extend(packaged)
        packaged = self._visit_to_package()
        if packaged:
            files.extend(packaged)
        return files

    @abc.abstractmethod
    def _visit_to_resolve(self):
        if type(self.data) is str and self.data[0] == '#':
            found = self.query_archive_object(self.data[1:])
            if not found:
                raise RuntimeError('not found hash tag: tag={}'.format(self.data))
            if isinstance(found, ArchiveList) or isinstance(found, ArchiveDict):
                self.data = found
            else:
                self.data = found.data

    @abc.abstractmethod
    def _visit_to_compose(self):
        pass
    
    @abc.abstractmethod
    def _visit_to_package(self):
        pass

    @abc.abstractmethod
    def compose(self):
        self.visit_to_compose_recursive()

    @abc.abstractmethod
    def package(self):
        files = self.visit_to_package_recursive()

    @classmethod
    def register_creator(cls, name, creator):
        ArchiveObject.factory[name] = creator

    @classmethod
    def parse_recursive(cls, obj):
        if not isinstance(obj, ArchiveObject):
            raise RuntimeError('not instance of ArchiveObject: obj={}'.format(obj))

        if type(obj.value) is dict:
            obj.data = {}
            for k, v in obj.value.items():
                data = None
                for name, creator in ArchiveObject.factory.items():
                    if name in k:
                        words = k.split('@')
                        data = creator(type=words[1], parent=obj, name=words[0], value=v)
                        if data:
                            k = words[0]
                            break
                if not data:
                    if type(v) is bool:
                        data = ArchiveBool(parent=obj, data=v)
                    elif type(v) is int:
                        data = ArchiveInt(parent=obj, data=v)
                    elif type(v) is str:
                        data = ArchiveStr(parent=obj, data=v)
                    elif type(v) is list:
                        data = ArchiveList(parent=obj, value=v)
                    elif type(v) is dict:
                        data = ArchiveDict(parent=obj, value=v)
                    elif not v:
                        data = ArchiveObject(parent=obj)
                    else:
                        raise RuntimeError('not support to create archive object: obj={}, key={}, value={}'.format(obj, k, v))
                obj.data[k] = data
        elif type(obj.value) is list:
            obj.data = []
            for v in obj.value:
                if type(v) is bool:
                    data = ArchiveBool(parent=obj, data=v)
                elif type(v) is int:
                    data = ArchiveInt(parent=obj, data=v)
                elif type(v) is str:
                    data = ArchiveStr(parent=obj, data=v)
                elif type(v) is list:
                    data = ArchiveList(parent=obj, value=v)
                elif type(v) is dict:
                    data = ArchiveDict(parent=obj, value=v)
                else:
                    raise RuntimeError('not support to create archive object: obj={}, v={}'.format(obj, v))
                obj.data.append(data)
        elif type(obj.value) in [bool, int, str]:
            raise RuntimeError('not support to create archive object: obj={}, obj.value={}'.format(obj, obj.value))
        return obj
    
    @classmethod
    def dump(cls, obj):
        pobj = None
        if isinstance(obj, ArchiveBool) or isinstance(obj, ArchiveInt) or isinstance(obj, ArchiveStr):
            pobj = obj.data
        elif isinstance(obj, ArchiveList):
            pobj = []
            for data in obj.data:
                pobj.append(ArchiveObject.dump(data))
        elif isinstance(obj, ArchiveDict):
            pobj = {}
            for k, v in obj.data.items():
                pobj[k] = ArchiveObject.dump(v)
        return pobj


class ArchiveBool(ArchiveObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ArchiveInt(ArchiveObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ArchiveStr(ArchiveObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ArchiveList(ArchiveObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ArchiveDict(ArchiveObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Archive(ArchiveObject):
    def __init__(self, data, filename=str()):
        super().__init__(name='root', value=data)
        self._filename = abspath(filename) if abspath(filename) else str()
        self._work_dir = '/'.join(['.', '_package_', str(time.time())])
        self.visit_to_resolve_recursive()

    @property
    def filename(self):
        return self._filename

    @property
    def work_dir(self):
        return self._work_dir

    @abc.abstractmethod
    def package(self):
        if not exists(self._work_dir):
            makedirs(self._work_dir)
        cwd = getcwd()
        chdir(self._work_dir)

        files = self.visit_to_package_recursive()
        for file in files:
            if exists(file) == False:
                raise RuntimeError('not found file: file={}'.format(file))

        if 'zip' in self._filename:
            with zipfile.ZipFile(self._filename, 'w') as zfile:
                for file in files:
                    zfile.write(file)
        else:
            with tarfile.open(self._filename, 'w') as tfile:
                for file in files:
                    tfile.add(file)

        chdir(cwd)


class ArchivePlugin(ArchiveObject):
    def __init__(self, **kwargs):
        super().__init__(manual_parse=True, **kwargs)
