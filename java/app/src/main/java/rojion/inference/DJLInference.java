package rojion.inference;

import java.awt.image.BufferedImage;
import java.io.IOException;

import org.json.simple.JSONObject;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

public class DJLInference extends Inference{
    public static final String NAME = "DJL";

    public DJLInference(JSONObject params) {
        super(params);
        //TODO Auto-generated constructor stub
    }

    public Object infer() throws ModelNotFoundException, MalformedModelException, IOException, TranslateException{
        String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
        BufferedImage img = BufferedImageUtils.fromUrl(url);

        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optFilter("backbone", "resnet50")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                System.out.println(detection);
            }
        }
        return null;
    }
    
}
