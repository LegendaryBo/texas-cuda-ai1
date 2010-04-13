package pl.wroc.uni.ii.evolution.distribution.workers.evalApplet;

import java.awt.Color;

import javax.swing.BorderFactory;
import javax.swing.JLabel;

/**
 * @author Kacper Gorski Simple component displaying current action in color
 */
@SuppressWarnings("serial")
public class EvActionDisplay extends JLabel {

  // 1 - waiting for job
  // 2 - evaluating
  // 3 - benchmarking
  // 4 - downloading individuals
  // 5 - uploading individuals
  public void setAction(int action) {
    if (action == 0) {
      setBackground(Color.WHITE);
      setText("        Connecting");
    }
    if (action == 1) {
      setBackground(Color.RED);
      setText("            Waiting");
    }
    if (action == 2) {
      setBackground(Color.GREEN);
      setText("         Evaluating");
    }
    if (action == 3) {
      setBackground(Color.BLUE);
      setText("     Benchmarking");
    }
    if (action == 4) {
      setBackground(Color.ORANGE);
      setText("   Downloading data");
    }
    if (action == 5) {
      setBackground(Color.PINK);
      setText("     Uploading result");
    }

    repaint();
  }


  public EvActionDisplay() {
    setText("Evaluating...");
    setOpaque(true);
    setBorder(BorderFactory.createBevelBorder(0));
    setAction(0);

  }

}
