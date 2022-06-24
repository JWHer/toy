package rojion.inference;

import org.junit.jupiter.api.Test;

import ai.onnxruntime.OrtException;

public class InferenceTest {
    @Test void inference() {
        Inference inferenceTest = new Inference();
        try {
			inferenceTest.infer(null);
		} catch (OrtException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
