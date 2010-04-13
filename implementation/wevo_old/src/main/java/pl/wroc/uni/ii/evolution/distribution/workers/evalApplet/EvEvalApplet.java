package pl.wroc.uni.ii.evolution.distribution.workers.evalApplet;

import java.awt.Color;
import java.awt.Font;
import java.util.Date;

import javax.swing.JApplet;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import pl.wroc.uni.ii.evolution.distribution.workers.EvEvalTaskInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvJARCacheImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoader;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoaderImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskMaster;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunicationImpl;

/**
 * @author Kacper Gorski
 */
@SuppressWarnings("serial")
public class EvEvalApplet extends JApplet implements EvEvalTaskInterface {

  // COMPONENTS
  private EvEvalWorkerTitle aplet_title = new EvEvalWorkerTitle();

  private JLabel action_label = new JLabel("Action:");

  private JLabel task_id_label = new JLabel("task id:");

  private JLabel work_id_label = new JLabel("work id:");

  private JLabel node_id_label = new JLabel("node id:");

  private JLabel objective_function = new JLabel("obj. function:");

  private JLabel progress_label = new JLabel("Progress:");

  private JLabel statistics_label = new JLabel("Statistics");

  private JLabel black_line = new JLabel();

  private JLabel black_line2 = new JLabel();

  private JLabel evaluated_individuals_label =
      new JLabel("Evaluated individuals: 0");

  private JLabel evaluated_time_label =
      new JLabel("Time spend on evaluation: 0 ms");

  private JLabel waiting_time_label = new JLabel("Time spend waiting: 0 ms");

  private JLabel download_time_label =
      new JLabel("Time spend downloading: 0 ms");

  private JLabel upload_time_label = new JLabel("Time spend uploading: 0 ms");

  private JLabel average_evaluation_time_label =
      new JLabel("Average evaluation time: 0 ms");

  private JLabel total_jobs_label = new JLabel("Total jobs completed: 0 ");

  private JLabel total_jobs_aborted = new JLabel("Total jobs aborted: 0 ");

  private JLabel console_label = new JLabel("Console:");

  private JTextArea console = new JTextArea();

  private JScrollPane console_pane = new JScrollPane(console);

  private EvProgressBar progress_bar = new EvProgressBar(400, 20);

  private EvActionDisplay action_display = new EvActionDisplay();

  private EvApletLogo aplet_logo = new EvApletLogo();

  private int evaluation_time;

  private int waiting_time;

  private int individual_evaluated;

  private int jobs;

  private int aborted;

  private int download_time;

  private int upload_time;

  private int completed;


  // END OF COMPONENTS

  public void init() {
    // positioning components
    setLayout(null);
    Font normal_font = action_label.getFont().deriveFont(0);
    aplet_title.setBounds(10, 10, 200, 60);
    add(aplet_title);
    aplet_logo.setBounds(200, 10, 400, 60);
    add(aplet_logo);
    action_label.setBounds(10, 80, 40, 20);
    add(action_label);
    action_label.setFont(normal_font);
    action_display.setBounds(50, 80, 120, 20);
    add(action_display);
    progress_label.setFont(normal_font);
    progress_label.setBounds(190, 80, 70, 20);
    add(progress_label);
    progress_bar.setBounds(250, 80, 400, 20);
    add(progress_bar);
    task_id_label.setBounds(10, 100, 80, 20);
    add(task_id_label);
    task_id_label.setFont(normal_font);
    work_id_label.setBounds(90, 100, 80, 20);
    add(work_id_label);
    work_id_label.setFont(normal_font);
    node_id_label.setBounds(170, 100, 200, 20);
    add(node_id_label);
    node_id_label.setFont(normal_font);
    objective_function.setBounds(370, 100, 300, 20);
    add(objective_function);
    objective_function.setFont(normal_font);
    black_line.setBounds(10, 120, 630, 1);
    add(black_line);
    black_line.setOpaque(true);
    black_line.setBackground(Color.BLACK);
    statistics_label.setBounds(250, 122, 100, 20);
    add(statistics_label);
    evaluated_individuals_label.setBounds(10, 142, 180, 20);
    add(evaluated_individuals_label);
    evaluated_individuals_label.setFont(normal_font);
    evaluated_time_label.setBounds(190, 142, 230, 20);
    add(evaluated_time_label);
    evaluated_time_label.setFont(normal_font);
    average_evaluation_time_label.setBounds(420, 142, 270, 20);
    add(average_evaluation_time_label);
    average_evaluation_time_label.setFont(normal_font);
    waiting_time_label.setBounds(10, 162, 180, 20);
    add(waiting_time_label);
    waiting_time_label.setFont(normal_font);
    upload_time_label.setBounds(190, 162, 230, 20);
    add(upload_time_label);
    upload_time_label.setFont(normal_font);
    download_time_label.setBounds(420, 162, 270, 20);
    add(download_time_label);
    download_time_label.setFont(normal_font);
    total_jobs_label.setBounds(10, 182, 200, 20);
    add(total_jobs_label);
    total_jobs_label.setFont(normal_font);
    total_jobs_aborted.setBounds(190, 182, 200, 20);
    add(total_jobs_aborted);
    total_jobs_aborted.setFont(normal_font);
    black_line2.setBounds(10, 202, 630, 1);
    add(black_line2);
    black_line2.setOpaque(true);
    black_line2.setBackground(Color.BLACK);
    console_label.setBounds(260, 204, 100, 20);
    add(console_label);
    console_pane.setBounds(22, 224, 600, 300);
    add(console_pane);
    console.setText("Objective function worker started");
    console.setEditable(false);
    // end of positioning components

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
        new EvTaskMaster(proxy, loader, jar_manager, 2000, 1, this);

    /** start interaction with managment servlet */
    interaction.start();

  }


  public void addEvaluationTime(int miliseconds) {
    evaluation_time += miliseconds;
    evaluated_time_label.setText("Time spend on evaluation: "
        + formatTime(evaluation_time));
    average_evaluation_time_label
        .setText("Average evaluation time: "
            + round(((double) evaluation_time)
                / ((double) individual_evaluated), 3) + " ms");
    consolePrintln("Job's evaluation time: " + formatTime(miliseconds));
  }


  public void addwaitingTime(int miliseconds) {
    waiting_time += miliseconds;
    waiting_time_label.setText("Time spend waiting: "
        + formatTime(waiting_time));
  }


  public void currentCellID(long cell_id) {
    // cell_id_label.setText("cell id: "+cell_id);
  }


  public void currentNodeID(long node_id) {
    node_id_label.setText("node id: " + node_id);
  }


  public void currentObjectiveFunction(String objFun) {
    objective_function.setText("obj. function: " + objFun);
  }


  public void currentState(int state) {
    if (state == 3)
      consolePrintln("Started benchmark");
    if (state == 1)
      consolePrintln("Waiting for job");
    if (state == 2)
      consolePrintln("Started evaluating");
    if (state == 4)
      consolePrintln("Downloading data");
    if (state == 5)
      consolePrintln("Uploading data");
    action_display.setAction(state);
  }


  public void currentTaskID(int task_id) {
    task_id_label.setText("task id: " + task_id);
  }


  public void individualEvaluated() {
    individual_evaluated++;
    progress_bar.addCounter();
    evaluated_individuals_label.setText("Evaluated individuals: "
        + individual_evaluated);
    if (individual_evaluated == completed) {
      jobFinished();
    }
  }


  public void jobFinished() {
    jobs++;
    total_jobs_label.setText("Total jobs completed: " + jobs);
    progress_bar.setNewJob(0);
    consolePrintln("Job finished");
  }


  public void addDownloadTime(int miliseconds) {
    download_time += miliseconds;
    download_time_label.setText("Time spend downloading: "
        + formatTime(download_time));
    consolePrintln(completed + " individuals downloaded in "
        + formatTime(miliseconds));
  }


  public void addUploadTime(int miliseconds) {
    upload_time += miliseconds;
    upload_time_label.setText("Time spend uploading: "
        + formatTime(upload_time));
    consolePrintln("results uploaded in " + formatTime(miliseconds));
  }


  public void newJobSize(int length) {
    completed = length;
    progress_bar.setNewJob(length);
    consolePrintln("New job of: " + length + " individuals started");
  }


  public void jobAborted() {
    aborted++;
    progress_bar.setNewJob(0);
    total_jobs_aborted.setText("Total jobs aborted: " + aborted);
    consolePrintln("Job aborted. It was completed by other evaluation node");
  }


  public static double round(double value, int decimalPlace) {
    double power_of_ten = 1;
    while (decimalPlace-- > 0)
      power_of_ten *= 10.0;
    return Math.round(value * power_of_ten) / power_of_ten;
  }


  private String formatTime(int time) {
    if (time < 1000)
      return time + " ms";
    if (time < 100000)
      return (time / 1000) + "." + (time % 1000) + " sec";
    return (time / 1000) + "." + ((time / (100)) % 10) + " sec";
  }


  @SuppressWarnings("deprecation")
  private void consolePrintln(String str) {
    Date date = new Date();
    String clock =
        date.getHours() + ":" + date.getMinutes() + ":" + date.getSeconds();
    console.setText(console.getText() + "\n" + clock + " " + str);
  }


  public void currentWorkID(long work_id) {
    work_id_label.setText("work id: " + work_id);

  }

}
