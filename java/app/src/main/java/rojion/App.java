/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package rojion;

public class App {
    public static final String VERSION = "0.0.1";
    public static final String APP_NAME = "ROJION";
    public static final String DESCRIPTION = "Real-time Onnx Java Inference ONline";

    public String getGreeting() {
        return APP_NAME + " - " + VERSION + "\n" + DESCRIPTION;
    }

    public static void main(String[] args) {
        System.out.println(new App().getGreeting());
    }
}