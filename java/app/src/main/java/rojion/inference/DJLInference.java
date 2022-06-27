package rojion.inference;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.json.simple.JSONObject;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
// import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;

public class DJLInference extends Inference {
    public static final String NAME = "DJL";

    public DJLInference(JSONObject params) throws IllegalArgumentException, IllegalAccessException {
        super(params);
        // TODO Auto-generated constructor stub
    }

    protected static void saveBoundingBoxImage(
            Image img, DetectedObjects detection, String outputDir, String outputName, String outputType)
            throws IOException {
        img.drawBoundingBoxes(detection);

        Path outputPath = Paths.get(outputDir);
        Files.createDirectories(outputPath);
        Path imagePath = outputPath.resolve(outputName);
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), outputType);
    }

    // public Object infer() throws ModelNotFoundException, MalformedModelException,
    // IOException, TranslateException {
    // String url =
    // "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
    // ImageFactory fac = ImageFactory.getInstance();
    // BufferedImage img = (BufferedImage) fac.fromUrl(url);

    // Criteria<BufferedImage, DetectedObjects> criteria = Criteria.builder()
    // .optApplication(Application.CV.OBJECT_DETECTION)
    // .setTypes(BufferedImage.class, DetectedObjects.class)
    // .optFilter("backbone", "resnet50")
    // .optProgress(new ProgressBar())
    // .build();

    // try (ZooModel<BufferedImage, DetectedObjects> model =
    // ModelZoo.loadModel(criteria)) {
    // try (Predictor<BufferedImage, DetectedObjects> predictor =
    // model.newPredictor()) {
    // DetectedObjects detection = predictor.predict(img);
    // System.out.println(detection);
    // }
    // }
    // return null;
    // }

    /*
     * 
     * DJL Model Builder
     * 
     */
    protected static class ModelBuilder {
        protected final List<String> classes;
        protected final String modelName;
        protected final String engine;
        protected String modelUrls;
        protected int imageSize = 640;
        protected float threshold = 0.3f;

        private Pipeline imagePipeline() {
            Pipeline pipeline = new Pipeline();
            pipeline.add(new Resize(imageSize));
            pipeline.add(new ToTensor());
            return pipeline;
        }

        protected Translator<Image, DetectedObjects> translator(Pipeline pipeline) {
            throw new UnsupportedOperationException("Abstract method translator not implemented");
        }

        private Criteria<Image, DetectedObjects> criteria(Translator<Image, DetectedObjects> translator) {
            return Criteria.builder()
                    .setTypes(Image.class, DetectedObjects.class)
                    .optModelUrls(modelUrls)
                    .optModelName(modelName)
                    .optTranslator(translator)
                    .optProgress(new ProgressBar())
                    .optEngine(engine)
                    .build();
        }

        public ModelBuilder(List<String> classes, String modelName, String engine) {
            this.classes = classes;
            this.modelName = modelName;
            this.engine = engine;
        }

        public ModelBuilder modelUrls(String modelUrls) {
            this.modelUrls = modelUrls;
            return this;
        }

        public ModelBuilder imageSize(int imageSize) {
            this.imageSize = imageSize;
            return this;
        }

        public ModelBuilder threshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        public ZooModel<Image, DetectedObjects> build()
                throws ModelNotFoundException, MalformedModelException, IOException {
            Pipeline pipeline = imagePipeline();
            Translator<Image, DetectedObjects> translator = translator(pipeline);
            Criteria<Image, DetectedObjects> criteria = criteria(translator);
            return criteria.loadModel();
        }
    }
}
