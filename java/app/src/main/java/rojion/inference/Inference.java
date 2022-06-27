package rojion.inference;

import java.util.List;
import java.util.logging.Logger;

import org.json.simple.JSONObject;

public abstract class Inference {
  public static final String NAME = "DEFAULT";
  protected Logger logger = Logger.getLogger("ROJION");
  protected JSONObject params;
  protected List<String> classes;
  protected String modelUrls;
  protected String modelName;
  protected String engine;
  protected int gpuDeviceId = -1;
  protected float threshold = 0.3f;

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
