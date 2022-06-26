package rojion.inference;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.json.simple.JSONObject;

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
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;

public class YoloInference extends Inference{

    public YoloInference(JSONObject params) {
        super(params);
        //TODO Auto-generated constructor stub
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
        throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        img.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected.png");
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
    }

    public Object infer(){
        int imageSize = 640;
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(imageSize));
        pipeline.add(new ToTensor());

        List<String> synset = new ArrayList<>(80);
        for (int i = 0; i < 80; i++) {
            synset.add("Person");
        }

        Translator<Image, DetectedObjects> translator =  YoloV5Translator
            .builder()
            .setPipeline(pipeline)
            .optSynset(synset)
            .optThreshold(0.8f)
            .build();

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
            .setTypes(Image.class, DetectedObjects.class)
            .optModelUrls("yolo/")
            .optModelName("best.pt")
            .optTranslator(translator)
            .optProgress(new ProgressBar())
            .optEngine("PyTorch")
            .build();

        try(ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                Image input = ImageFactory.getInstance().fromUrl("https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg");
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
            }
        }
    }
    
}
