package pl.wroc.uni.ii.evolution.distribution.statistics.applet.statTypes;

import java.io.IOException;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JTextArea;
import javax.swing.JTextField;

import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvGenesAverageValuesChart;
import pl.wroc.uni.ii.evolution.distribution.statistics.applet.EvDistributetStatisticsApplet;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;

/**
 * @author Kacper Gorski
 */
public class EvAvgGenesValueStatPanel extends EvStatisticPanel {

  private static final long serialVersionUID = 1L;

  private JLabel tip =
      new JLabel(
          "Indexes of genes to be displayed divided by comma: (Ex. 1,4,13,20)");

  private JTextArea indexes = new JTextArea();

  private JButton ok_button = new JButton("OK");

  private JLabel start_iteration_label =
      new JLabel("Starting iteration:(leave blank to select all)");

  private JTextField start_iteration_text = new JTextField();

  private JLabel end_iteration_label = new JLabel("Ending iteration:");

  private JTextField end_iteration_text = new JTextField();

  private JCheckBox logarithmic = new JCheckBox(" - logarythmic scale");


  public EvAvgGenesValueStatPanel(EvDistributetStatisticsApplet applet) {
    this.applet = applet;

    setLayout(null);

    tip.setBounds(0, 0, 400, 20);
    add(tip);
    indexes.setBounds(0, 20, 300, 40);
    add(indexes);

    start_iteration_label.setBounds(0, 60, 300, 20);
    add(start_iteration_label);
    start_iteration_text.setBounds(0, 80, 100, 20);
    add(start_iteration_text);
    end_iteration_label.setBounds(0, 100, 300, 20);
    add(end_iteration_label);
    end_iteration_text.setBounds(0, 120, 100, 20);
    add(end_iteration_text);

    ok_button.setBounds(100, 170, 100, 20);
    add(ok_button);

    logarithmic.setBounds(0, 140, 200, 20);
    add(logarithmic);

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
            EvStatisticFilter.byClass(EvGenesAvgValueStatistic.class, stat_tab);
      else
        stat_tab =
            EvStatisticFilter.byClassAndIteration(
                EvGenesAvgValueStatistic.class, stat_tab, starting_iteration,
                ending_iteration);
    } catch (IOException e) {
      e.printStackTrace();
    }

    int[] indexes = getSelectedIndexes();

    // selecting all indexes
    if (getSelectedIndexes() == null) {
      int ind_size =
          ((EvGenesAvgValueStatistic) stat_tab[0]).genes_value.length;
      indexes = new int[ind_size];
      for (int i = 0; i < ind_size; i++)
        indexes[i] = i;
    }

    return EvGenesAverageValuesChart.createJFreeChart(stat_tab, logarithmic
        .isSelected(), indexes);
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


  private int[] getSelectedIndexes() {
    String input = indexes.getText();

    if (input.equals("")) {
      return null;
    } else {

      String[] tab_string = input.split(",");
      int[] tab_int = new int[tab_string.length];

      for (int i = 0; i < tab_string.length; i++) {
        tab_int[i] = Integer.parseInt(tab_string[i]);
      }

      return tab_int;
    }
  }


  // statistic name
  public String getName() {
    return "Average value of the genes";
  }

}
