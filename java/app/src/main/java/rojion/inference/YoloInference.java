package rojion.inference;

import java.io.IOException;
import java.nio.file.Files;
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
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

public class YoloInference extends Inference{
    public YoloInference(JSONObject params) {
        super(params);
        
        setDefaults();
    }
    private void setDefaults(){
        classes = Arrays.asList(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        );
        modelUrls = "models/";
        modelName = "yolov5s.torchscript";
        engine = "Pytorch";
        threshold = 0.3f;
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection) throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        img.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected2.png");
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
    }

    private Pipeline imagePipeline(int imageSize){
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(imageSize));
        pipeline.add(new ToTensor());
        return pipeline;
    }
    private Pipeline imagePipeline(){
        return imagePipeline(640);
    }

    private Translator<Image, DetectedObjects> yoloV5Translator(Pipeline pipeline, List<String> synonymSet, float threshold){
        return YoloV5Translator.builder()
            .setPipeline(pipeline)
            .optSynset(synonymSet)
            .optThreshold(threshold)
            .build();
    }
    private Translator<Image, DetectedObjects> yoloV5Translator(Pipeline pipeline, List<String> synonymSet){
        return yoloV5Translator(pipeline, synonymSet, threshold);
    }
    private Translator<Image, DetectedObjects> yoloV5Translator(Pipeline pipeline){
        return yoloV5Translator(pipeline, classes);
    }
    private Translator<Image, DetectedObjects> yoloV5Translator(List<String> synonymSet){
        return yoloV5Translator(imagePipeline(), synonymSet);
    }
    private Translator<Image, DetectedObjects> yoloV5Translator(){
        return yoloV5Translator(imagePipeline(), classes);
    }

    private Criteria<Image, DetectedObjects> criteria(Translator<Image, DetectedObjects> translator){
        return Criteria.builder()
            .setTypes(Image.class, DetectedObjects.class)
            .optModelUrls(modelUrls)
            .optModelName(modelName)
            .optTranslator(translator)
            .optProgress(new ProgressBar())
            .optEngine(engine)
            .build();
    }
    private Criteria<Image, DetectedObjects> criteria(){
        return criteria(yoloV5Translator());
    }

    public Object infer() throws ModelNotFoundException, MalformedModelException, IOException, TranslateException{
        int imageSize = 640;
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(imageSize));
        pipeline.add(new ToTensor());

        List<String> synset = Arrays.asList(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        );

        Translator<Image, DetectedObjects> translator =  YoloV5Translator
            .builder()
            .setPipeline(pipeline)
            .optSynset(synset)
            .optThreshold(0.3f)
            .build();

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
            .setTypes(Image.class, DetectedObjects.class)
            .optModelUrls("models/")
            .optModelName("yolov5s.torchscript")
            .optTranslator(translator)
            .optProgress(new ProgressBar())
            .optEngine("PyTorch")
            .build();
            
        try(ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                // Image input = ImageFactory.getInstance().fromUrl("https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg");
                Image input = ImageFactory.getInstance().fromUrl("file:///home/jwher/사진/val2017/000000000139.jpg");
                DetectedObjects objects = predictor.predict(input);
                List<BoundingBox> boxes = new ArrayList<>();
                List<String> names = new ArrayList<>();
                List<Double> prob = new ArrayList<>();
                for (Classifications.Classification obj : objects.items()) {
                    DetectedObjects.DetectedObject objConvered = (DetectedObjects.DetectedObject) obj;
                    BoundingBox box = objConvered.getBoundingBox();
                    Rectangle rec = box.getBounds();
                    Rectangle rec2 = new Rectangle(
                        rec.getX() / 640,
                        rec.getY() / 640,
                        rec.getWidth() / 640,
                        rec.getHeight() / 640
                        );
                    boxes.add(rec2);
                    names.add(obj.getClassName());
                    prob.add(obj.getProbability());
                }
                DetectedObjects converted = new DetectedObjects(names, prob, boxes);
                saveBoundingBoxImage(input, converted);
                return converted;
            }
        }
        
        // return null;
    }
    
}
