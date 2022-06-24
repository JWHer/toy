package rojion.inference;

import java.util.Map;
import java.util.logging.Logger;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import javax.imageio.ImageIO;

public class Inference {
    private static final Logger logger = Logger.getLogger("ROJION");
    private final int gpuDeviceId = 0;
    private String modelPath = ".models/yolov5s.onnx";

    public void infer(Map<String, OnnxTensor> inputs) throws OrtException {
        var env = OrtEnvironment.getEnvironment();
        var sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.addCUDA(gpuDeviceId);
        var session = env.createSession("model.onnx", sessionOptions);
        showModelInfos(session);
        
        var result = session.run(inputs);
        // return result;
    }

    private void showModelInfos(OrtSession session) throws OrtException{
        logger.info("Inputs:");
        for (NodeInfo i : session.getInputInfo().values()) {
          logger.info(i.toString());
        }

        logger.info("Outputs:");
        for (NodeInfo i : session.getOutputInfo().values()) {
          logger.info(i.toString());
        }
    }

    public Map<String, OnnxTensor> loadImage(String path) {
        return null;
    }
}
