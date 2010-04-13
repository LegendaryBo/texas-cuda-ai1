package pl.wroc.uni.ii.evolution.engine.operators.general.display;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

import javax.swing.JFrame;
import javax.swing.JPanel;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Displays a graph of basic information about a population on a given JPanel
 * 
 * @author Marcin Brodziak
 */
public class EvRealtimeOnJPanelStatistics<T extends EvIndividual> implements
    EvOperator<T> {

  public static <T extends EvIndividual> EvRealtimeOnJPanelStatistics<T> displayFrame(
      double min_y, double max_y) {
    JFrame frame = new JFrame();
    frame.setSize(300, 300);
    JPanel panel = new JPanel();
    frame.add(panel);
    EvRealtimeOnJPanelStatistics<T> stats =
        new EvRealtimeOnJPanelStatistics<T>(panel, min_y, max_y);
    frame.setVisible(true);
    return stats;
  }

  JPanel graph;

  double MAX_Y, MIN_Y;

  ArrayList<Double> min_l, max_l, avg_l, stddev_l;

  private EvGatherStatistics<T> stats = new EvGatherStatistics<T>();


  /**
   * @param graph1 JPanel object on which statistscs are painted
   * @param min_y - minimal value on the diagram
   * @param max_y - not uset now
   */
  public EvRealtimeOnJPanelStatistics(JPanel graph1, double min_y, double max_y) {
    graph = graph1;
    MIN_Y = min_y;
    MAX_Y = Double.MIN_VALUE;
    min_l = new ArrayList<Double>();
    max_l = new ArrayList<Double>();
    avg_l = new ArrayList<Double>();
    stddev_l = new ArrayList<Double>();
  }


  public EvPopulation<T> apply(EvPopulation<T> population) {
    computeCurrentParams(population);
    BufferedImage b =
        new BufferedImage(graph.getWidth(), graph.getHeight(),
            BufferedImage.TYPE_INT_RGB);
    drawParams(max_l, b.getGraphics(), Color.GREEN);
    drawParams(min_l, b.getGraphics(), Color.RED);
    drawParams(avg_l, b.getGraphics(), Color.BLUE);
    drawParams(stddev_l, b.getGraphics(), Color.MAGENTA);
    Graphics g = b.getGraphics();
    g.setColor(Color.WHITE);
    g.drawString("current_best: " + max_l.get(max_l.size() - 1).toString(), 0,
        20);
    g.drawString("best: " + MAX_Y, 0, 40);
    graph.getGraphics().drawImage(b, 0, 0, b.getWidth(), b.getHeight(), null);
    return population;
  }


  private void drawParams(ArrayList<Double> list, Graphics g, Color color) {
    int x = 0;
    int p_y = 0;
    g.setColor(color);
    for (Double max : list) {
      int y =
          (int) (graph.getHeight() - max * graph.getHeight() / (MAX_Y - MIN_Y));
      if (x > 0)
        g.drawLine(x, p_y, x + 1, y);
      p_y = y;
      x++;
    }
  }


  private void computeCurrentParams(EvPopulation<T> population) {
    stats.apply(population);
    min_l.add(stats.lastMin());
    avg_l.add(stats.lastMean());
    stddev_l.add(stats.lastStD());

    double max = stats.lastMax();
    max_l.add(max);
    if (max > MAX_Y)
      MAX_Y = max;
  }
}