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
        Class<? extends Inference> clazz = this.getClass();
        for (Object keyObj : params.keySet()) {
            String key = (String) keyObj;
            try {
                Field field = getField(clazz, key);
                field.set(this, params.get(key));
            } catch (NoSuchFieldException e) {
                logger.warning("field: [" + key + "] not exist");
                e.printStackTrace();
            } catch (SecurityException e) {
                logger.severe("Security exception!");
                e.printStackTrace();
                throw e;
            }
        }
    }

    private static Field getField(Class clazz, String fieldName)
            throws NoSuchFieldException {
        try {
            return clazz.getDeclaredField(fieldName);
        } catch (NoSuchFieldException e) {
            Class<? extends Inference> superClass = clazz.getSuperclass();
            if (superClass == null) {
                throw e;
            } else {
                return getField(superClass, fieldName);
            }
        }
    }

    public DetectedObjects infer(String imageUrl) throws Exception {
        throw new UnsupportedOperationException("Abstract method infer not implemented");
    }

}
