package rojion.inference;

import java.util.logging.Logger;

import org.json.simple.JSONObject;

// import ai.onnxruntime.NodeInfo;
// import ai.onnxruntime.OnnxTensor;
// import ai.onnxruntime.OrtEnvironment;
// import ai.onnxruntime.OrtException;
// import ai.onnxruntime.OrtSession;

public class OnnxInference extends Inference {
    public OnnxInference(JSONObject params) throws IllegalArgumentException, IllegalAccessException {
        super(params);
    }
 
    private String modelPath = ".models/yolov5s.onnx";

    // public void infer(Map<String, OnnxTensor> inputs) throws OrtException {
    //     var env = OrtEnvironment.getEnvironment();
    //     var sessionOptions = new OrtSession.SessionOptions();
    //     sessionOptions.addCUDA(gpuDeviceId);
    //     var session = env.createSession("model.onnx", sessionOptions);
    //     showModelInfos(session);
        
    //     var result = session.run(inputs);
    //     // return result;
    // }

    // private void showModelInfos(OrtSession session) throws OrtException{
    //     logger.info("Inputs:");
    //     for (NodeInfo i : session.getInputInfo().values()) {
    //       logger.info(i.toString());
    //     }

    //     logger.info("Outputs:");
    //     for (NodeInfo i : session.getOutputInfo().values()) {
    //       logger.info(i.toString());
    //     }
    // }

    // public Map<String, OnnxTensor> loadImage(String path) {
    //     return null;
    // }
}
