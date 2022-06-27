package rojion.json;

import org.json.simple.JSONObject;
import org.junit.Test;

import junit.framework.TestCase;

public class jsonSimpleTest extends TestCase{

    private JSONObject jsonObject = new JSONObject();

    @Override
    protected void setUp() {
        jsonObject.put("a", 1);
    }

    @Test
    public void testGet() {
        Object value = jsonObject.get("a");
        System.out.println(value);

        // throw error or return null
        Object value2 = jsonObject.get("b");
        System.out.println(value2);
        // => return null
    }

    @Override
    protected void tearDown() {

    }
}
