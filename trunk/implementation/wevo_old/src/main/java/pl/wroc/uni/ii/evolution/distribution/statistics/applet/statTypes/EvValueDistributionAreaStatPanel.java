package pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes;

import java.io.IOException;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JTextField;

import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvObjectiveFunctionValueDistributionAreaChart;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.EvDistributetStatisticsApplet;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveFunctionValueDistributionStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * @author Kacper Gorski
 */
public class EvValueDistributionAreaStatPanel extends EvStatisticPanel {

  private static final long serialVersionUID = 1L;

  private JLabel title_start_iteration =
      new JLabel("Starting iteration:(leave blank to select all)");

  private JLabel title_end_iteration = new JLabel("Ending iteration:");

  private JTextField start_iteration_text = new JTextField();

  private JTextField end_iteration_text = new JTextField();

  private JLabel bins_title = new JLabel("Number of bins:");

  private JTextField bins_text = new JTextField();

  private JButton ok_butt = new JButton("OK");


  public EvValueDistributionAreaStatPanel(EvDistributetStatisticsApplet applet) {
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

    bins_title.setBounds(0, 80, 200, 20);
    add(bins_title);

    bins_text.setBounds(0, 100, 200, 20);
    add(bins_text);

    ok_butt.setBounds(0, 120, 100, 20);
    add(ok_butt);

    ok_butt.addActionListener(this);

  }


  // if fields are blank, return 0
  private int getEndingIteration() {
    if (start_iteration_text.getText().equals(""))
      return 0;
    else
      return Integer.parseInt(end_iteration_text.getText());
  }


  // if field left blank, return 0
  private int getStartingIteration() {
    if (start_iteration_text.getText().equals(""))
      return 0;
    else
      return Integer.parseInt(start_iteration_text.getText());

  }


  @Override
  public JFreeChart generate(int task_id, Long[] cell_id, long node_id,
      EvDBServletCommunication data_base) {
    int bins = Integer.parseInt(bins_text.getText());
    int starting_iteration = getStartingIteration();
    int ending_iteration = getEndingIteration();

    EvStatistic[] stat_tab = null;
    try {
      stat_tab = getStatistics(task_id, cell_id[0], node_id, data_base);
      if (ending_iteration == 0)
        stat_tab =
            EvStatisticFilter.byClass(
                EvObjectiveFunctionValueDistributionStatistic.class, stat_tab);
      else
        stat_tab =
            EvStatisticFilter.byClassAndIteration(
                EvObjectiveFunctionValueDistributionStatistic.class, stat_tab,
                starting_iteration, ending_iteration);

    } catch (IOException e) {
      e.printStackTrace();
    }

    return EvObjectiveFunctionValueDistributionAreaChart.createChart(stat_tab,
        bins);
  }


  @Override
  public String getName() {
    return "Value distribution area";
  }

}
