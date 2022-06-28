package rojion.webcam;

import java.util.logging.Logger;
import java.awt.image.BufferedImage;

import javax.swing.JOptionPane;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.translate.TranslateException;
import nu.pattern.OpenCV;

public class WebCam {
    protected Logger logger = Logger.getLogger("ROJION");

    public WebCam() {
    }

    public void live(Predictor<Image, DetectedObjects> predictor) throws TranslateException {
        // System.out.println(Core.getBuildInformation());
        // OpenCV.loadShared();
        // OpenCV.loadLocally();
        // System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // System.loadLibrary("opencv_videoio_ffmpeg455_64");
        // System.setProperty("java.library.path", "/lib64");
        // System.loadLibrary("opencv_ffmpeg300_64");

        // VideoCapture capture = new VideoCapture(0);
        VideoCapture capture = new VideoCapture();
        // "rtsp://admin:snuai3883!@192.168.0.249:554/0/onvif/profile5/media.smp"
        // "rtsp://admin:init123!!@192.168.0.207:7001/6e03a7c5-e172-7eab-2e44-1b12a48aeed4"
        // "rtsp://admin:init123!!@192.168.0.207:7001/3fc9c921-2562-2252-7128-3dc2ec049d87"
        capture.open("output/test.avi");
        // try {
        //     System.out.println("Delay waiting..");
        //     Thread.sleep(10000); // wait while stream open from dvr
        //     System.out.println("Delay end..");
        // } catch(InterruptedException ex) {
        //     Thread.currentThread().interrupt();
        // }

        if (!capture.isOpened()){
            logger.warning("No camera detected");
            return;
        }

        Mat image = new Mat();
        boolean captured = false;
        // .5초간 10번 캡쳐 시도, 한번이라도 성공하면
        for (int i = 0; i < 10; ++i) {
            captured = capture.read(image);
            if (captured) {
                break;
            }
            // try {
            //     Thread.sleep(50);
            // } catch (InterruptedException ignore) {
            //     // ignore
            // }
        }
        if (!captured) {
            JOptionPane.showConfirmDialog(null, "Failed to capture image from WebCam.");
        }

        ViewerFrame frame = new ViewerFrame(image.width(), image.height());
        ImageFactory factory = ImageFactory.getInstance();

        while (capture.isOpened()) {
            if (!capture.read(image)) {
                break;
            }
            Image img = factory.fromImage(image);
            DetectedObjects detections = predictor.predict(img);
            img.drawBoundingBoxes(detections);

            frame.showImage(toBufferedImage((Mat) img.getWrappedImage()));
        }

        capture.release();
        predictor.close();
    }

    private static BufferedImage toBufferedImage(Mat mat) {
        int width = mat.width();
        int height = mat.height();
        int type =
                mat.channels() != 1 ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;

        if (type == BufferedImage.TYPE_3BYTE_BGR) {
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);
        }

        byte[] data = new byte[width * height * (int) mat.elemSize()];
        mat.get(0, 0, data);

        BufferedImage ret = new BufferedImage(width, height, type);
        ret.getRaster().setDataElements(0, 0, width, height, data);

        return ret;
    }
}
