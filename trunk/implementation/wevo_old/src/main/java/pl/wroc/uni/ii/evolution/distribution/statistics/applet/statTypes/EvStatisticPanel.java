package pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.JPanel;
import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.distribution.statistics.applet.EvChartFrame;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.EvDistributetStatisticsApplet;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * Abstract class which represents the Panel on which you can define properties
 * of the statistics
 * 
 * @author Kacper Gorski
 */
public abstract class EvStatisticPanel extends JPanel implements ActionListener {

  protected EvDistributetStatisticsApplet applet;


  // generates stats using the given data
  public abstract JFreeChart generate(int task_id, Long[] cell_id,
      long node_id, EvDBServletCommunication data_base);


  // returns the name of statistics (like average genes statistics)
  public abstract String getName();


  // called when 'ok' pressed
  public void actionPerformed(ActionEvent e) {

    int task_id = applet.task_selector.getTaskId();
    Long[] cell_id = applet.domain_selector.getCellId();
    long node_id = applet.domain_selector.getNodeId();

    JFreeChart chart = generate(task_id, cell_id, node_id, applet.db_comm);

    JFrame frame = new EvChartFrame(chart);
    frame.setBounds(200, 200, 500, 400);
    frame.setVisible(true);

  }


  // returns EvStatistics tab of the given param.
  public EvStatistic[] getStatistics(int task_id, long cell_id, long node_id,
      EvDBServletCommunication data_base) throws IOException {

    EvStatistic[] stat_tab = null;
    stat_tab = data_base.getStatistics(task_id, cell_id, node_id);
    return stat_tab;
  }

}
