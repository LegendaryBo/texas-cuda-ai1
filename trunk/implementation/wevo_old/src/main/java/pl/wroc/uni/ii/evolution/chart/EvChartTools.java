package pl.wroc.uni.ii.evolution.chart;

import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import javax.swing.JPanel;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;

import com.sun.image.codec.jpeg.JPEGCodec;
import com.sun.image.codec.jpeg.JPEGEncodeParam;
import com.sun.image.codec.jpeg.JPEGImageEncoder;

/**
 * Class containing static functions for creating final products of
 * visualization process. For every JFreeChart object user can:
 * <ul>
 * <li> create JPEG file
 * <li> create BufferedImage object
 * </ul>
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public final class EvChartTools {

  /**
   * Disabling default constructor.
   */
  private EvChartTools() {

  }


  /**
   * Creates <b>JPEG file </b> for given JFreeChart chart. A few properties of
   * image can be adjusted.
   * 
   * @param chart -- input chart
   * @param path -- path of a result JPEG file
   * @param height -- height of JPEG image
   * @param width -- width of JPEG image
   * @param quality -- quality of JPEG image
   * @throws IOException -- if some error occurred during encoding image to JPEG
   *         format
   */
  public static void createJPG(final JFreeChart chart, final String path,
      final int height, final int width, final float quality)
      throws IOException {

    BufferedImage img = draw(chart, width, height);
    FileOutputStream fos = new FileOutputStream(path);
    JPEGImageEncoder encoder2 = JPEGCodec.createJPEGEncoder(fos);

    JPEGEncodeParam param2 = encoder2.getDefaultJPEGEncodeParam(img);
    param2.setQuality((float) quality, true);
    encoder2.encode(img, param2);

    fos.close();
  }


  /**
   * Creates <b> BufferedImage </b> for given JFreeChart chart.
   * 
   * @param chart -- input chart
   * @param width -- width of BufferedImage
   * @param height -- height of BufferedImage
   * @return BufferedImage with chart
   */
  public static BufferedImage draw(final JFreeChart chart, final int width,
      final int height) {

    BufferedImage img =
        new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    Graphics2D g2 = img.createGraphics();

    chart.draw(g2, new Rectangle2D.Double(0, 0, width, height));
    g2.dispose();
    return img;
  }


  /**
   * Writes <b> BufferedImage </b> to <b>OutputStream</b>.
   * 
   * @param out -- any OutputStream
   * @param image -- any BufferedImage
   * @throws IOException -- if some error occurred during encoding image to JPEG
   *         format
   */
  public static void writeToStream(final OutputStream out,
      final BufferedImage image) throws IOException {

    ChartUtilities.writeBufferedImageAsJPEG(out, image);
  }


  /**
   * Fast conversion from JFreeChart into JPanel object.
   * 
   * @param chart to be displayed in JPanel
   * @return JPanel object containing interactive JFreeChart
   */
  public static JPanel freeChartToJPanel(final JFreeChart chart) {
    final ChartPanel panel = new ChartPanel(chart, true);
    return panel;
  }

}
