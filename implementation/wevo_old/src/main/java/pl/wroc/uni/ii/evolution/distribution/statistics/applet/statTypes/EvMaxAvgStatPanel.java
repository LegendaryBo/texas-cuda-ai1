package pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes;

import java.io.IOException;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JTextField;

import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvObjectiveFunctionValueMaxAvgMinChart;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.EvDistributetStatisticsApplet;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * @author Kacper Gorski
 */
public class EvMaxAvgStatPanel extends EvStatisticPanel {

  private static final long serialVersionUID = 1L;

  private JLabel title_start_iteration =
      new JLabel("Starting iteration:(leave blank to select all)");

  private JLabel title_end_iteration = new JLabel("Ending iteration:");

  private JTextField start_iteration_text = new JTextField();

  private JTextField end_iteration_text = new JTextField();

  private JButton ok_butt = new JButton("OK");

  private JCheckBox logarithmic = new JCheckBox(" - logarythmic scale");


  public EvMaxAvgStatPanel(EvDistributetStatisticsApplet applet) {
    this.applet = applet;
    setLayout(null);

    title_start_iteration.setBounds(0, 0, 300, 20);
    add(title_start_iteration);
    start_iteration_text.setBounds(0, 20, 100, 20);
    add(start_iteration_text);

    title_end_iteration.setBounds(0, 40, 300, 20);
    add(title_end_iteration);
    end_iteration_text.setBounds(0, 60, 100, 20);
    add(end_iteration_text);

    ok_butt.setBounds(0, 110, 100, 20);
    add(ok_butt);
    ok_butt.addActionListener(this);

    logarithmic.setBounds(0, 80, 200, 20);
    add(logarithmic);
  }


  @Override
  public JFreeChart generate(int task_id, Long[] cell_id, long node_id,
      EvDBServletCommunication data_base) {
    int starting_iteration = getStartingIteration();
    int ending_iteration = getEndingIteration();

    EvStatistic[] stat_tab = null;
    try {
      stat_tab = getStatistics(task_id, cell_id[0], node_id, data_base);
      if (ending_iteration == 0)
        stat_tab =
            EvStatisticFilter.byClass(
                EvObjectiveFunctionValueMaxAvgMinStatistic.class, stat_tab);
      else
        stat_tab =
            EvStatisticFilter.byClassAndIteration(
                EvObjectiveFunctionValueMaxAvgMinStatistic.class, stat_tab,
                starting_iteration, ending_iteration);
    } catch (IOException e) {
      e.printStackTrace();
    }

    return EvObjectiveFunctionValueMaxAvgMinChart.createJFreeChart(stat_tab,
        logarithmic.isSelected());

  }


  private int getEndingIteration() {
    if (start_iteration_text.getText().equals(""))
      return 0;
    else
      return Integer.parseInt(end_iteration_text.getText());
  }


  private int getStartingIteration() {
    if (start_iteration_text.getText().equals(""))
      return 0;
    else
      return Integer.parseInt(start_iteration_text.getText());

  }


  // title
  public String getName() {
    return "Max avg min statistic";
  }

}
