package pl.wroc.uni.ii.evolution.distribution.workers.evalApplet;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JPanel;

/**
 * @author Kacper Gorski simple progress bar used to show progress of evaluation
 *         of job in eval aplet
 */
@SuppressWarnings("serial")
public class EvProgressBar extends JPanel {

  private int width = 0;

  private int height = 0;

  private int total = 0;

  private int completed = 0;


  public EvProgressBar(int width, int height) {
    this.width = width;
    this.height = height;
  }


  // resets bar
  public void setNewJob(int ind_number) {
    total = ind_number;
    completed = 0;
    repaint();
  }


  // add one individual
  public void addCounter() {
    completed++;
    repaint();
  }


  // overwrite
  public void paint(Graphics g) {
    super.paint(g);
    g.setColor(Color.CYAN);
    if (completed != 0)
      g.fillRect(0, 0, (int) (width * ((double) completed / (double) total)),
          height);
    g.draw3DRect(0, 0, width - 1, height - 1, false);
    g.setColor(Color.DARK_GRAY);
    g.drawRect(0, 0, width, height);
    g.setColor(Color.BLACK);
    g.drawString("finished " + completed + " of " + total, 150, 14);
  }

}
