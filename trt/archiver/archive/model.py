from archive.common import create_plugin_package, ArchiveDict, ArchiveObject

import abc
from os.path import exists
import tensorrt
from urllib.parse import urlparse
import logging

class Model(ArchiveDict):
    def __init__(self, type=str(), **kwargs):
        super().__init__(**kwargs)
        self._type = type if type else str()

    @property
    def type(self):
        return self._type

    @classmethod
    def create(cls, type=str(), **kwargs):
        words = type.split('.')
        if 'tensorrt' == words[1]:
            return TensorRTModel(**kwargs)
        elif 'onnx' == words[1]:
            return OnnxModel(**kwargs)

class TensorRTModel(Model):
    def __init__(self, **kwargs):
        super().__init__(type='tensorrt', **kwargs)
        self._logger = None
        self._builder = None
        self._config = None
        self._network = None
        self._profile = None
        self.initialize()

    @property
    def logger(self):
        return self._logger

    @property
    def builder(self):
        return self._builder

    @property
    def config(self):
        return self._config

    @property
    def network(self):
        return self._network

    @property
    def profile(self):
        return self._profile

    def initialize(self):
        self._logger = tensorrt.Logger()
        self._logger.min_severity = tensorrt.Logger.Severity.VERBOSE
        self._builder = tensorrt.Builder(self._logger)
        self._config = self._builder.create_builder_config()
        self._network = self._builder.create_network(flags=1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self._profile = self._builder.create_optimization_profile()
        self._plugin = {}
        
        dumped = ArchiveObject.dump(self)
        for k, v in dumped.items():
            if k == 'plugin':
                self._plugin = v
                for name, kwargs in self._plugin.items():
                    if kwargs:
                        self._plugin[name] = create_plugin_package('model', name, parent=self, **kwargs)
                    else:
                        self._plugin[name] = create_plugin_package('model', name, parent=self)

            elif hasattr(self, '_' + k):
                obj = getattr(self, '_' + k)
                for kk, vv in v.items():
                    if kk == 'max_batch_size':
                        obj.max_batch_size = vv
                        
                    elif kk == 'max_workspace_size':
                        obj.max_workspace_size = vv * 1 << 30 # (vv GiB)

                    elif kk == 'flag':
                        if vv == 1: # FP16
                            if not self._builder.platform_has_fast_fp16:
                                print("[Archiver] FP16 not supported on this platform.")
                            else:
                                obj.set_flag(tensorrt.BuilderFlag.FP16)
                                print("[Archiver] FP16 precision is successfully adapted")

                    elif kk == 'dynamic_batch_size':
                        self.dynamic_batch_size = vv

    def mark_outputs(self, network):
        last_layer = network.get_layer(network.num_layers - 1)
        if not last_layer.num_outputs:
            print("Last layer contains no outputs.")
            return
        for i in range(last_layer.num_outputs):
            network.mark_output(last_layer.get_output(i))

    def check_network(self, network):
        if not network.num_outputs:
            print("[Archiver] No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
            self.mark_outputs(network)
        
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

        print("[Archiver] === Network Description ===")
        for i, inp in enumerate(inputs):
            print("[Archiver] Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))

        for i, out in enumerate(outputs):
            print("[Archiver] Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))

    def load_from(self, source):
        dumped = ArchiveObject.dump(source)
        parsed = urlparse(dumped['url'])
        if parsed.scheme != 'file' or exists(parsed.path) == False:
            raise RuntimeError('failure in loading from source: source={}'.format(dumped))
        if source.type == 'onnx':
            parser = tensorrt.OnnxParser(self._network, self._logger)

            success = parser.parse_from_file(parsed.path)

            for idx in range(parser.num_errors):
                print(parser.get_error(idx))

            if not success:
                pass # Error handling code here in the future
            else:
                print("[Archiver] Success in parsing onnx model")
    
    def set_dynamic_profile(self):
        if self.dynamic_batch_size[0] != self.dynamic_batch_size[1] != self.dynamic_batch_size[2]:
            self.check_network(self._network)
            min_batch_size = self.dynamic_batch_size[0]
            opt_batch_size = self.dynamic_batch_size[1]
            max_batch_size = self.dynamic_batch_size[2]

            # Archiver Function Addition #1.1 - make optimization profile
            inp_name = self._network.get_input(0).name
            inp_shape = self._network.get_input(0).shape[1:]

            self._profile.set_shape(inp_name, (min_batch_size, *inp_shape), (opt_batch_size, *inp_shape), (max_batch_size, *inp_shape))
            self._config.add_optimization_profile(self._profile)
            print("[Archiver] Dynamic batched engine is successfully built")

        else: # Fixed batch size 
            self.check_network(self._network)
            min_batch_size = 1
            opt_batch_size = 1
            max_batch_size = 1

            # Archiver Function Addition #1.1 - make optimization profile
            inp_name = self._network.get_input(0).name
            inp_shape = self._network.get_input(0).shape[1:]

            self._profile.set_shape(inp_name, (min_batch_size, *inp_shape), (opt_batch_size, *inp_shape), (max_batch_size, *inp_shape))
            self._config.add_optimization_profile(self._profile)
            print("[Archiver] Fixed batched engine is successfully built")

    def save(self):
        dumped = ArchiveObject.dump(self)
        parsed = urlparse(dumped['url'])

        if parsed.scheme != 'file':
            raise RuntimeError('failure in saving at url: url={}'.format(dumped['url']))
        if exists(parsed.path) == True:
            print('overwrite file: url={}'.format(dumped['url']))

        # Serialized된 engine을 저장하는 일반적인 방법으로 수정하였습니다.
        # engine = self._builder.build_engine(self._network, self._config)
        # with open(parsed.path, 'wb') as file:
        #     file.write(engine.serialize())

        serialized_engine = self._builder.build_serialized_network(self._network, self._config)
        with open(parsed.path, 'wb') as file:
            file.write(serialized_engine)

        return parsed.path

    @abc.abstractmethod
    def _visit_to_compose(self):
        self.load_from(self.data['from'].data)
        self.set_dynamic_profile()
        for plugin in self._plugin.values():
            plugin._visit_to_compose()
        self.save()
        
    @abc.abstractmethod
    def _visit_to_package(self):
        pass

class OnnxModel(Model):
    def __init__(self, **kwargs):
        super().__init__(type='onnx', **kwargs)


if __name__ != '__main__':
    ArchiveObject.register_creator('@model', Model.create)
