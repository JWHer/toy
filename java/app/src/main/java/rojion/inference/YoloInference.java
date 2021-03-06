package rojion.inference;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.json.simple.JSONObject;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

public class YoloInference extends DJLInference {
    public YoloInference(JSONObject params) throws IllegalArgumentException, IllegalAccessException {
        super(params);
    }

    @Override
    protected void setDefaults() {
        super.setDefaults();
        classes = Arrays.asList(
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear",
                "hair drier", "toothbrush");
        modelName = "yolov5s.torchscript";
        engine = "PyTorch";
        imageSize = 640;
    }

    @Override
    public ZooModel<Image, DetectedObjects> getModel()
            throws ModelNotFoundException, MalformedModelException, IOException {
        if (model == null) {
            model = new ModelBuilder(classes, modelName, engine)
                    .modelUrls(modelUrls).imageSize(imageSize).threshold(threshold)
                    .build();
        }
        return model;
    }

    @Override
    public DetectedObjects infer(String imageUrl)
            throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
        // ZooModel<Image, DetectedObjects> model = new ModelBuilder(classes, modelName, engine)
        //     .modelUrls(modelUrls).imageSize(imageSize).threshold(threshold)
        //     .build();
        // Predictor<Image, DetectedObjects> predictor = model.newPredictor();
        Predictor<Image, DetectedObjects> predictor = getPredictor();

        // ImageFactory.getInstance().fromUrl("https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg");
        Image input = ImageFactory.getInstance().fromUrl(imageUrl);
        Path imagePath = Paths.get(imageUrl);
        String outputName = imagePath.getFileName().toString();
        int extPos = outputName.lastIndexOf('.');
        String outputType = "";
        if (extPos != -1)
            outputType = outputName.substring(extPos + 1);
        else
            outputType = saveImageType;

        // predictor.batchPredict(inputs)
        DetectedObjects objects = predictor.predict(input);
        DetectedObjects converted = convYoloBox(objects);
        saveBoundingBoxImage(input, converted, outputDir, outputName, outputType);

        // predictor.close();
        // model.close();
        return converted;
    }

    public DetectedObjects convYoloBox(DetectedObjects objects){
        List<BoundingBox> boxes = new ArrayList<>();
        List<String> names = new ArrayList<>();
        List<Double> prob = new ArrayList<>();
        for (Classifications.Classification obj : objects.items()) {
            DetectedObjects.DetectedObject objConvered = (DetectedObjects.DetectedObject) obj;
            BoundingBox box = objConvered.getBoundingBox();
            Rectangle rec = box.getBounds();
            Rectangle rec2 = new Rectangle(
                    rec.getX() / imageSize,
                    rec.getY() / imageSize,
                    rec.getWidth() / imageSize,
                    rec.getHeight() / imageSize);
            boxes.add(rec2);
            names.add(obj.getClassName());
            prob.add(obj.getProbability());
        }
        return new DetectedObjects(names, prob, boxes);
    }

    protected static class ModelBuilder extends DJLInference.ModelBuilder {
        public ModelBuilder(List<String> classes, String modelName, String engine) {
            super(classes, modelName, engine);
        }

        @Override
        protected Translator<Image, DetectedObjects> translator(Pipeline pipeline) {
            return YoloV5Translator.builder()
                    .setPipeline(pipeline)
                    .optSynset(classes)
                    .optThreshold(threshold)
                    .build();
        }
    }
}
