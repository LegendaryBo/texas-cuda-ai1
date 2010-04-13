package pl.wroc.uni.ii.evolution.distribution.statistics.applet;

import javax.swing.JFrame;
import javax.swing.JScrollPane;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;

/**
 * @author Kacper Gorski
 */
public class EvChartFrame extends JFrame {

  private static final long serialVersionUID = 4112598891223941857L;


  public EvChartFrame(JFreeChart chart) {
    final ChartPanel chartPanel = new ChartPanel(chart);
    final JScrollPane scroll = new JScrollPane(chartPanel);
    scroll.setPreferredSize(new java.awt.Dimension(500, 270));
    add(scroll);
  }

}
