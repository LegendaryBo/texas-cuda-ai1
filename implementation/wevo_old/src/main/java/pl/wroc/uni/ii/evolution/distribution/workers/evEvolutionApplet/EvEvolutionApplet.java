package pl.wroc.uni.ii.evolution.distribution.workers.evEvolutionApplet;

import java.awt.Color;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Date;
import java.util.Hashtable;
import java.util.List;

import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import pl.wroc.uni.ii.evolution.distribution.statistics.persistency.EvOperationalMemoryStatisticStorage;
import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvJARCacheImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoader;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoaderImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskMaster;
import pl.wroc.uni.ii.evolution.distribution.workers.evalApplet.EvApletLogo;
import pl.wroc.uni.ii.evolution.distribution.workers.evalApplet.EvEvalApplet;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunicationImpl;

/**
 * @author Kacper Gorski Main applet class which performs evolutionary algorithm
 *         based on EvIsland model. It show many interesting informations about
 *         evolution, charts, CPU usage, etc...
 */
@SuppressWarnings("serial")
public class EvEvolutionApplet extends JApplet implements EvEvolutionInterface,
    ActionListener {

  // Components
  private EvEvolAppletLogo aplet_logo = new EvEvolAppletLogo();

  private EvApletLogo wevo_logo = new EvApletLogo();

  private JLabel iteration = new JLabel("Iteration: ");

  private JLabel total_iterations_label = new JLabel("total iterations: ");

  private JLabel task_id = new JLabel("Task_id: ");

  private JLabel cell_id = new JLabel("Cell_id: ");

  private JLabel node_id = new JLabel("Node_id: ");

  private JLabel best_individual = new JLabel("Best individual: ");

  private JLabel best_individual_value = new JLabel("Best individual value: ");

  private JLabel population_size = new JLabel("Population size: ");

  private JLabel total_time = new JLabel("Total time: ");

  private JLabel avg_time_per_iter = new JLabel("Average time per iteration: ");

  private JLabel details_pop_label =
      new JLabel("Show details of population recently: ");

  private JButton evaluated_details = new JButton("Evaluated");

  private JButton imported_details = new JButton("Imported");

  private JButton exported_details = new JButton("Exported");

  private EvChartPanel chart_panel = new EvChartPanel();

  private JLabel operators_label = new JLabel("Operators aplify time:");

  private JTextArea operators_time = new JTextArea();

  private JScrollPane operators_pane = new JScrollPane(operators_time);

  private JTextArea console = new JTextArea();

  private JLabel console_label = new JLabel("Console:");

  private JScrollPane console_pane = new JScrollPane(console);

  private JLabel black_line = new JLabel();

  private JLabel black_line2 = new JLabel();

  private JLabel black_line3 = new JLabel();

  private EvOperationalMemoryStatisticStorage stat_storage =
      new EvOperationalMemoryStatisticStorage();

  private EvObjectiveFunctionValueMaxAvgMinGatherer<?> gatherer =
      new EvObjectiveFunctionValueMaxAvgMinGatherer(stat_storage);

  private int current_iteration;

  private int total_iterations;

  private EvPopulation<?> population = null;

  private EvPopulation<?> exported_population = null;

  private EvPopulation<?> imported_population = null;

  private int time_counter;

  private int total_operators_time = 0;

  private Hashtable<EvOperator, Integer> operators =
      new Hashtable<EvOperator, Integer>();


  public void init() {
    setLayout(null);
    Font normal_font = task_id.getFont().deriveFont(0);

    aplet_logo.setBounds(10, 10, 200, 60);
    add(aplet_logo);
    aplet_logo.setFont(normal_font);
    wevo_logo.setBounds(200, 10, 400, 60);
    add(wevo_logo);
    wevo_logo.setFont(normal_font);
    task_id.setBounds(10, 70, 100, 20);
    add(task_id);
    task_id.setFont(normal_font);
    cell_id.setBounds(110, 70, 100, 20);
    add(cell_id);
    cell_id.setFont(normal_font);
    node_id.setBounds(210, 70, 150, 20);
    add(node_id);
    node_id.setFont(normal_font);
    iteration.setBounds(410, 70, 100, 20);
    add(iteration);
    iteration.setFont(normal_font);
    total_iterations_label.setBounds(510, 70, 150, 20);
    add(total_iterations_label);
    total_iterations_label.setFont(normal_font);

    black_line.setBounds(315, 95, 1, 310);
    add(black_line);
    black_line.setOpaque(true);
    black_line.setBackground(Color.BLACK);
    black_line2.setBounds(10, 95, 600, 1);
    add(black_line2);
    black_line2.setOpaque(true);
    black_line2.setBackground(Color.BLACK);
    black_line3.setBounds(10, 405, 600, 1);
    add(black_line3);
    black_line3.setOpaque(true);
    black_line3.setBackground(Color.BLACK);
    chart_panel.setBounds(320, 100, 300, 300);
    add(chart_panel);

    best_individual.setBounds(10, 100, 300, 20);
    add(best_individual);
    best_individual.setFont(normal_font);
    best_individual_value.setBounds(10, 120, 300, 20);
    add(best_individual_value);
    best_individual_value.setFont(normal_font);
    population_size.setBounds(10, 140, 300, 20);
    add(population_size);
    population_size.setFont(normal_font);
    total_time.setBounds(10, 160, 300, 20);
    add(total_time);
    total_time.setFont(normal_font);
    avg_time_per_iter.setBounds(10, 180, 300, 20);
    add(avg_time_per_iter);
    avg_time_per_iter.setFont(normal_font);

    details_pop_label.setBounds(60, 210, 300, 20);
    add(details_pop_label);
    details_pop_label.setFont(normal_font);
    evaluated_details.setBounds(10, 230, 100, 20);
    add(evaluated_details);
    evaluated_details.addActionListener(this);
    imported_details.setBounds(110, 230, 100, 20);
    add(imported_details);
    imported_details.addActionListener(this);
    exported_details.setBounds(210, 230, 100, 20);
    add(exported_details);
    exported_details.addActionListener(this);

    operators_label.setBounds(80, 260, 150, 20);
    add(operators_label);
    operators_label.setFont(normal_font);

    operators_pane.setBounds(10, 280, 300, 120);
    add(operators_pane);

    console_label.setBounds(270, 410, 100, 20);
    add(console_label);
    console_label.setFont(normal_font);
    console_pane.setBounds(10, 430, 610, 150);
    add(console_pane);
    console.setEditable(false);

    // String wevo_server_url = "http://192.168.0.6:8080/";
    // EvManagmentServletCommunication proxy = new
    // EvManagmentServletCommunicationImpl("http://192.168.0.6:8080/wevo_system/DistributionManager");

    String wevo_server_url = getParameter("wevo_server_url");
    EvManagmentServletCommunication proxy =
        new EvManagmentServletCommunicationImpl(
            getParameter("managment_servlet"));

    EvTaskLoader loader = new EvTaskLoaderImpl(wevo_server_url);
    EvJARCacheImpl jar_manager = new EvJARCacheImpl(proxy);
    jar_manager.init(System.currentTimeMillis());

    EvTaskMaster interaction =
        new EvTaskMaster(proxy, loader, jar_manager, 2000, 0, this);

    /** start interaction with managment servlet */
    interaction.start();

  }


  public void addEvalTime(int time) {
    time_counter += time;
    total_time.setText("Total time: " + formatTime(time_counter));
    avg_time_per_iter.setText("Average time per iteration: "
        + formatTime(time_counter / current_iteration));
  }


  public void addExportedPopulation(List population) {
    exported_population = new EvPopulation(population);
    consolePrintln("Exported population of " + population.size()
        + " individuals to server");
  }


  public void addImportedPopulation(List population) {
    consolePrintln("Imported population of " + population.size()
        + " individuals from server");
  }


  public void addOperatortime(EvOperator operator, int time) {
    total_operators_time += time;
    consolePrintln("Applying " + operator.getClass().getSimpleName());
    if (operators.containsKey(operator)) {
      System.out.println("jest");
      operators.put(operator, operators.get(operator) + time);
    } else {
      System.out.println("ni ma");
      operators.put(operator, time);
    }
    writeOperators();
  }


  private void writeOperators() {
    String txt = new String();
    if (operators.size() == 0)
      txt = "No data";
    else {

      for (EvOperator i : operators.keySet()) {
        int time = operators.get(i);
        double percent = 0.0;
        if (total_operators_time != 0)
          percent =
              EvEvalApplet.round((double) time / total_operators_time * 100, 2);
        txt +=
            i.getClass().getSimpleName() + " - " + formatTime(time) + " ("
                + percent + "%)\n";
      }
    }
    operators_time.setText(txt);

  }


  @SuppressWarnings("deprecation")
  private void consolePrintln(String str) {
    Date date = new Date();
    String clock =
        date.getHours() + ":" + date.getMinutes() + ":" + date.getSeconds();
    console.setText(console.getText() + "" + clock + " " + str + "\n");
  }


  public void currentNode(int node_id) {
    this.node_id.setText("Node id: " + node_id);
    consolePrintln("Node id: " + node_id);
  }


  @SuppressWarnings("unchecked")
  public void currentPopulation(EvPopulation population) {
    best_individual_value.setText("Best individual value: "
        + population.getBestResult().getObjectiveFunctionValue());
    best_individual.setText("Best individual: " + population.getBestResult());
    this.population = population;
    gatherer.apply(population);
    chart_panel.setStats(stat_storage);
  }


  public void currentTask(int task_id) {
    this.task_id.setText("Task id: " + task_id);
    consolePrintln("Task id: " + task_id);
  }


  public void iterationProgress() {
    current_iteration++;
    total_iterations++;
    iteration.setText("Iteration: " + current_iteration);
    total_iterations_label.setText("Total iterations: " + total_iterations);
    consolePrintln("Starting " + current_iteration + " iteration");
  }


  public void populationSize(int size) {
    population_size.setText("Population size: " + size);
  }


  public void actionPerformed(ActionEvent e) {
    if (e.getSource() == evaluated_details) {
      EvPopulationViewer viewer =
          new EvPopulationViewer(population, "Population at iteration "
              + current_iteration);
      viewer.setBounds(0, 0, 650, 550);
      viewer.setVisible(true);
    }
    if (e.getSource() == imported_details) {
      EvPopulationViewer viewer =
          new EvPopulationViewer(imported_population,
              "Population recently imported at iteration " + current_iteration);
      viewer.setBounds(0, 0, 650, 550);
      viewer.setVisible(true);
    }
    if (e.getSource() == exported_details) {
      EvPopulationViewer viewer =
          new EvPopulationViewer(exported_population,
              "Population recently exported at iteration " + current_iteration);
      viewer.setBounds(0, 0, 650, 550);
      viewer.setVisible(true);
    }

  }


  private String formatTime(int time) {
    if (time < 1000)
      return time + " ms";
    if (time < 100000)
      return (time / 1000) + "." + (time % 1000) + " sec";
    return (time / 1000) + "." + ((time / (100)) % 10) + " sec";
  }


  public void newTask() {
    consolePrintln("Starting new task");
    iteration.setText("Iteration: " + 0);
    current_iteration = 0;
    operators = new Hashtable<EvOperator, Integer>();
    writeOperators();
  }


  public void currentCellID(long cell_id) {
    consolePrintln("Cell id: " + cell_id);
    this.cell_id.setText("Cell_id: " + cell_id);
  }

}
