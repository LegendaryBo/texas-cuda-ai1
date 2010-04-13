package pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes;

import java.io.IOException;
import java.util.ArrayList;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JRadioButton;

import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvDistributetChart;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.EvDistributetStatisticsApplet;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genesorigin.EvGenesOriginStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * @author Kacper Gorski Component that let user to create distribute statistics
 */
public class EvDistributedObjValueStat extends EvStatisticPanel {

  private static final long serialVersionUID = -8334302021391761925L;

  private JLabel tip = new JLabel("Select statistics type:");

  private JRadioButton all_radiobutton = new JRadioButton(" - Everything");

  private JRadioButton max_radiobutton =
      new JRadioButton(" - Best individuals only");

  private JRadioButton avg_radiobutton =
      new JRadioButton(" - Average population value only");

  private JRadioButton min_radiobutton =
      new JRadioButton(" - Worst individuals only");

  private JRadioButton deviation_radiobutton =
      new JRadioButton(" - Standard deviation only");

  private JRadioButton genes_origin_radiobutton =
      new JRadioButton(" - Genes origin");

  private ButtonGroup group = new ButtonGroup();

  private JButton ok_button = new JButton("OK");


  public EvDistributedObjValueStat(EvDistributetStatisticsApplet applet) {
    setLayout(null);
    tip.setBounds(0, 0, 200, 20);
    add(tip);
    all_radiobutton.setEnabled(true);
    group.add(all_radiobutton);
    group.add(max_radiobutton);
    group.add(avg_radiobutton);
    group.add(min_radiobutton);
    group.add(deviation_radiobutton);
    group.add(genes_origin_radiobutton);

    this.applet = applet;

    all_radiobutton.setBounds(0, 20, 300, 20);
    add(all_radiobutton);
    all_radiobutton.setSelected(true);
    max_radiobutton.setBounds(0, 40, 300, 20);
    add(max_radiobutton);
    avg_radiobutton.setBounds(0, 60, 300, 20);
    add(avg_radiobutton);
    min_radiobutton.setBounds(0, 80, 300, 20);
    add(min_radiobutton);
    deviation_radiobutton.setBounds(0, 100, 300, 20);
    add(deviation_radiobutton);
    genes_origin_radiobutton.setBounds(0, 120, 300, 20);
    add(genes_origin_radiobutton);
    ok_button.setBounds(0, 140, 100, 20);
    add(ok_button);

    ok_button.addActionListener(this);
  }


  @Override
  public JFreeChart generate(int task_id, Long[] cells, long node_id,
      EvDBServletCommunication data_base) {

    ArrayList<EvStatistic[]> stats = new ArrayList<EvStatistic[]>();

    for (int i = 0; i < cells.length; i++) {
      Long[] nodes = null;
      try {
        nodes = data_base.getNodesIdsWithStatistics(task_id, cells[i]);
      } catch (IOException e) {
        e.printStackTrace();
      }

      if (nodes != null) {
        for (int j = 0; j < nodes.length; j++) {
          try {
            if (!genes_origin_radiobutton.isSelected())
              stats.add(EvStatisticFilter.byClass(
                  EvObjectiveFunctionValueMaxAvgMinStatistic.class, data_base
                      .getStatistics(task_id, cells[i], nodes[j])));
            else
              stats.add(EvStatisticFilter.byClass(EvGenesOriginStatistic.class,
                  data_base.getStatistics(task_id, cells[i], nodes[j])));
          } catch (IOException e) {
            e.printStackTrace();
          }
        }
      }
    }

    if (max_radiobutton.isSelected())
      return EvDistributetChart.createJFreeChartWithMax(stats
          .toArray(new EvStatistic[stats.size()][]));
    if (avg_radiobutton.isSelected())
      return EvDistributetChart.createJFreeChartWithAvg(stats
          .toArray(new EvStatistic[stats.size()][]));
    if (deviation_radiobutton.isSelected())
      return EvDistributetChart.createJFreeChartWithStdDev(stats
          .toArray(new EvStatistic[stats.size()][]));
    if (min_radiobutton.isSelected())
      return EvDistributetChart.createJFreeChartWithMin(stats
          .toArray(new EvStatistic[stats.size()][]));
    if (all_radiobutton.isSelected())
      return EvDistributetChart.createJFreeChart(stats
          .toArray(new EvStatistic[stats.size()][]));
    if (genes_origin_radiobutton.isSelected())
      return EvDistributetChart.createGenesOriginJFreeChart(stats
          .toArray(new EvStatistic[stats.size()][]));

    return null;
  }


  @Override
  public String getName() {
    return "Distributed statistics";
  }

}
