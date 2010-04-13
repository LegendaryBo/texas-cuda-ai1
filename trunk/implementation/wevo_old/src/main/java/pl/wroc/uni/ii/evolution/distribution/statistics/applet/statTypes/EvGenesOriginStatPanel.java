package pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes;

import java.io.IOException;

import javax.swing.JButton;

import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvGenesOriginChart;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.EvDistributetStatisticsApplet;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genesorigin.EvGenesOriginStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * @author Kacper Gorski
 */
public class EvGenesOriginStatPanel extends EvStatisticPanel {

  private static final long serialVersionUID = 1L;

  private JButton ok_button = new JButton("OK");


  public EvGenesOriginStatPanel(EvDistributetStatisticsApplet applet) {
    this.applet = applet;

    setLayout(null);

    ok_button.setBounds(0, 0, 100, 20);
    add(ok_button);
    ok_button.addActionListener(this);
  }


  @Override
  public JFreeChart generate(int task_id, Long[] cell_id, long node_id,
      EvDBServletCommunication data_base) {

    EvStatistic[] stat_tab = null;
    try {
      stat_tab = getStatistics(task_id, cell_id[0], node_id, data_base);
      stat_tab =
          EvStatisticFilter.byClass(EvGenesOriginStatistic.class, stat_tab);
    } catch (IOException e) {
      e.printStackTrace();
    }

    return EvGenesOriginChart.createJFreeChart(stat_tab);
  }


  @Override
  public String getName() {
    return "Genes origin statistics";
  }

}
