package pl.wroc.uni.ii.evolution.distribution.workers.evalApplet;

import java.awt.Graphics;
import java.awt.Image;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JLabel;

/**
 * @author Kacper Gorski
 */
@SuppressWarnings("serial")
public class EvEvalWorkerTitle extends JLabel {
  private Image image = null;


  public EvEvalWorkerTitle() {

    try {
      image = ImageIO.read(getClass().getResource("worker.GIF"));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  public void paint(Graphics g) {
    super.paint(g);
    g.drawImage(image, 0, 0, this);
  }

}
