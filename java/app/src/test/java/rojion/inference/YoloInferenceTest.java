package rojion.inference;

import org.json.simple.JSONObject;
import org.junit.Test;

import junit.framework.TestCase;

public class YoloInferenceTest extends TestCase{

    @Test
    public void testInfer() throws Exception {
        JSONObject params = new JSONObject();
        params.put("threshold", 0.2f);

        var factory = new InferenceFactory();
        
        var inf = factory.create("YOLO", params);
        var result = inf.infer("https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg");
        System.out.println(result);
        
    }
    
}
