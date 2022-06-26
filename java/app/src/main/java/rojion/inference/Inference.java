package rojion.inference;

import java.util.logging.Logger;

import org.json.simple.JSONObject;

public abstract class Inference {
  public static final String NAME = "DEFAULT";
  private JSONObject params;
  private static final Logger logger = Logger.getLogger("ROJION");
  private final int gpuDeviceId = 0;

  public Inference(JSONObject params){
    this.params = params;
    // try {
    //   InferenceFactory.resister(this.getClass());
    // } catch (Exception e) {
    //   logger.warning("Cannot register class ["+this.NAME+"]");
    //   e.printStackTrace();
    // }
  }

  public Object infer() throws Exception{
    throw new UnsupportedOperationException("Abstract method infer not implemented");
  }

}
