package rojion.inference;

import org.json.simple.JSONObject;
import com.google.common.collect.ImmutableMap;

import java.util.Map;

public class InferenceFactory {
    private static Map<String, Class<? extends Inference>> infFactories;
    static {
        infFactories = ImmutableMap.of(
            "Torch", TorchInference.class,
            "DJL", DJLInference.class,
            "YOLO", YoloInference.class
        );
    }

    public Inference create(String name, JSONObject params) throws Exception{
        Class<? extends Inference> cls = InferenceFactory.infFactories.get(name);
        return cls.getConstructor(JSONObject.class).newInstance(params);
    }
}
