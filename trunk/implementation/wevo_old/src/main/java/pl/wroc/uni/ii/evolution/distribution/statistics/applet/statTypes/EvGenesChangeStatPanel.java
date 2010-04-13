package pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes;

import java.io.IOException;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JTextField;
import org.jfree.chart.JFreeChart;
import pl.wroc.uni.ii.evolution.chart.EvGenesChangesChart;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.EvDistributetStatisticsApplet;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genechange.EvGeneChangeStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * Container which contains components needed to create genes changes
 * statistics.
 * 
 * @author Kacper Gorski
 */
public class EvGenesChangeStatPanel extends EvStatisticPanel {

  private static final long serialVersionUID = 1L;

  private JButton ok_button = new JButton("OK");

  private JLabel start_iteration_label =
      new JLabel("Starting iteration:(leave blank to select all)");

  private JTextField start_iteration_text = new JTextField();

  private JLabel end_iteration_label = new JLabel("Ending iteration:");

  private JTextField end_iteration_text = new JTextField();

  private JCheckBox logarithmic = new JCheckBox(" - logarythmic scale");


  public EvGenesChangeStatPanel(EvDistributetStatisticsApplet applet) {
    this.applet = applet;

    setLayout(null);

    start_iteration_label.setBounds(0, 0, 300, 20);
    add(start_iteration_label);
    start_iteration_text.setBounds(0, 20, 100, 20);
    add(start_iteration_text);
    end_iteration_label.setBounds(0, 40, 300, 20);
    add(end_iteration_label);
    end_iteration_text.setBounds(0, 60, 100, 20);
    add(end_iteration_text);

    logarithmic.setBounds(0, 80, 200, 20);
    add(logarithmic);

    ok_button.setBounds(0, 100, 100, 20);
    add(ok_button);

    ok_button.addActionListener(this);
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
            EvStatisticFilter.byClass(EvGeneChangeStatistic.class, stat_tab);
      else
        stat_tab =
            EvStatisticFilter.byClassAndIteration(EvGeneChangeStatistic.class,
                stat_tab, starting_iteration, ending_iteration);
    } catch (IOException e) {
      e.printStackTrace();
    }

    return EvGenesChangesChart.createJFreeChart(stat_tab, logarithmic
        .isSelected());
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
  public String getName() {
    return "Genes changes statistics";
  }

}
