package rojion.inference;

import java.lang.reflect.Field;
import java.util.List;
import java.util.logging.Logger;

import org.json.simple.JSONObject;

import ai.djl.modality.cv.output.DetectedObjects;

public abstract class Inference {
    public static final String NAME = "DEFAULT";
    protected Logger logger = Logger.getLogger("ROJION");
    protected JSONObject params;
    protected List<String> classes;
    protected String modelUrls = "models/";
    protected String modelName;
    protected String outputDir = "output/";
    protected String engine;
    protected String saveImageType = "png";
    protected int gpuDeviceId = -1;
    protected float threshold = 0.3f;

    public Inference(JSONObject params) throws IllegalArgumentException, IllegalAccessException {
        this.params = params;
        setDefaults();
        setParams();
    }

    protected void setDefaults() {
    }

    protected void setParams() throws IllegalArgumentException, IllegalAccessException {
        for (Field field : this.getClass().getFields()) {
            String key = field.getName();
            Object value = params.get(key);
            if(value!=null) field.set(this, value);
        }
    }

    public DetectedObjects infer(String imageUrl) throws Exception {
        throw new UnsupportedOperationException("Abstract method infer not implemented");
    }

}
