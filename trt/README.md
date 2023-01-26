![header](https://capsule-render.vercel.app/api?type=cylinder&color=gradient&customColorList=1&text=Model%20Archiver&height=200&fontSize=56)

[A simple example of converting onnx to trt](examples/trt.py)

Usage:  
    --archive-file        : Onnx File to convert trt  
    --archive-config-file : Config file(.yaml) for trt

Examples
```
model:
  sources:
    infer@model.onnx:
      url: file://.archiver/resources/tmp/PersonNet_v1.1.0.onnx
      tuning:
        onnx@tuning:
          plugin:
            onnx_tuning:
              test_arg: test
    trt@model.tensorrt:
      from: "#model.sources.infer"
      url: file://.archiver/resources/tmp/PersonNet_v1.1.0.fp16.engine
      builder:
        max_batch_size: 1
      config:
        max_workspace_size: 3
        flag: 1
      profile:
        dynamic_batch_size: [1, 1, 1]
      plugin:
        custom_yolo_layer:
  runtime:
    framework:
      infer@framework.deepstream.nvinfer:
        configs:
          property:
            labelfile-path: file://.archiver/resources/tmp/labels.txt
            custom-lib-path: file://.archiver/resources/tmp/PeopleNet_v1.1.0.parser.so
            parse-bbox-func-name: NvDsInferParseTxModel
            model-engine-file: "#model.sources.trt.url"
            gpu-id : "0"
            net-scale-factor : "0.0039215697906911373"
            model-color-format : "0"
            num-detected-classes : "3"
            interval : "0"
            gie-unique-id : "1"
            batch-size : "1"
            network-mode : "2"
            process-mode : "1"
            network-type : "0"
            symmetric-padding : "1"
            infer-dims : "3;512;512"
            cluster-mode : "3"
            network-input-order : "0"
          class-attrs-all:
            threshold : "0.5"
            dbscan-min-score : "0.7"
            nms-iou-threshold : "0.7"
        attributes:
          class_type: PersonNet_v1.0.0
```
