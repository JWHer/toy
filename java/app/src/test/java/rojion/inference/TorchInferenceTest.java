package rojion.inference;

import org.junit.Test;

public class TorchInferenceTest {
    @Test void inference() {
        var inf = new TorchInference(null);
        inf.infer();
    }
}
