package rojion.inference;

import java.util.Arrays;

import org.json.simple.JSONObject;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class TorchInference extends Inference {
    public static final String NAME = "TORCH";
    private String modelPath = ".models/yolov5s.pt";

    public TorchInference(JSONObject params) {
        super(params);
        //TODO Auto-generated constructor stub
    }

    public Object infer(){
        Module mod = Module.load(this.modelPath);
        Tensor data = Tensor.fromBlob(
            new int[] {1, 2, 3, 4, 5, 6}, // data
            new long[] {2, 3} // shape
        );
        IValue result = mod.forward(IValue.from(data), IValue.from(3.0));
        Tensor output = result.toTensor();
        System.out.println("shape: " + Arrays.toString(output.shape()));
        System.out.println("data: " + Arrays.toString(output.getDataAsFloatArray()));

        return output;
    }
}
