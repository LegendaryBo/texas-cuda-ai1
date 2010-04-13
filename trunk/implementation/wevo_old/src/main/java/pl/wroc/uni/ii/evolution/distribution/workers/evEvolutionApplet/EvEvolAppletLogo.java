package pl.wroc.uni.ii.evolution.distribution.workers.evEvolutionApplet;

import java.awt.Graphics;
import java.awt.Image;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JPanel;

/**
 * @author Kacper Gorski
 */
@SuppressWarnings("serial")
public class EvEvolAppletLogo extends JPanel {
  private Image image = null;


  public EvEvolAppletLogo() {

    try {
      image = ImageIO.read(getClass().getResource("worker2.PNG"));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  public void paint(Graphics g) {
    super.paint(g);
    g.drawImage(image, 0, 0, this);
  }
}
