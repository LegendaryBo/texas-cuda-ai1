package pl.wroc.uni.ii.evolution.distribution.workers.console.worker;

import java.util.List;

import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvJARCacheImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoader;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoaderImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskMaster;
import pl.wroc.uni.ii.evolution.distribution.workers.console.EvConsoleParameters;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunicationImpl;

/**
 * Main class implementing console program used to distribute evaluation of
 * evolution algorithms in <B>island models</b>.<br>
 * The program connects to the wEvo framework and downloads tasks which re then
 * evaluated. Information about this process are redirected to standard output.<br>
 * It implements interface EvEvolutionInterface which tells the programs what's
 * going on inside the evolution.<br>
 * <br>
 * Parameters: <br>
 * <B>-s</B> <i>wevo_url</i> - connect to specified wevo_url. If it's not
 * specified the program will try to connect server specified in
 * <b>worker_options</b> file in current directory <br>
 * <b>-c</b> <i>option</i> (possible options are: <B>0</b> - turn off
 * reporting on the standard output, <B>1</b> - minor reporting, <B>2</b> -
 * average reporting, <B>3</b> - max reporting) <br>
 * <b>-r</b> <i>file</i> - (serialize best individual to a file named
 * "file(current_date)") <br>
 * <br>
 * <br>
 * Examples:<br>
 * <i>javac EvConsoleWorker -s http://127.0.0.1</i><br>
 * <I>javac EvConsoleWorker -s http://127.0.0.1</I> -c 1 -f best_result
 * 
 * @author Kacper Gorski
 */
public class EvConsoleWorker implements EvEvolutionInterface {

  public int reporting = 3;

  // name of the file where we put serialized best result
  public String output_individual_file;


  public static void main(String args[]) {

    EvConsoleWorker this_application = new EvConsoleWorker();
    EvConsoleParameters parameters = new EvConsoleParameters(args);

    // retrieve report intensivity
    if (parameters.parameterExist('c')) {
      this_application.reporting =
          Integer.parseInt(parameters.getParameter('c'));
    }

    // retrieve file name to serialize individual
    if (parameters.parameterExist('f')) {
      this_application.output_individual_file = parameters.getParameter('f');
    }

    // retrieve server address
    String wevo_server_url;
    if (!parameters.parameterExist('s')) {
      wevo_server_url = "http://127.0.0.1:8080/";
    } else {
      wevo_server_url = parameters.getParameter('s');
    }

    EvManagmentServletCommunication wevo_manager_server =
        new EvManagmentServletCommunicationImpl(wevo_server_url
            + "wevo_system/DistributionManager");

    EvTaskLoader loader = new EvTaskLoaderImpl(wevo_server_url);
    EvJARCacheImpl jar_manager = new EvJARCacheImpl(wevo_manager_server);
    // create temporary directory with current time as directory name
    jar_manager.init(System.currentTimeMillis());

    EvTaskMaster interaction =
        new EvTaskMaster(wevo_manager_server, loader, jar_manager, 2000, 0,
            this_application);

    /** start interaction with management servlet */
    // application will connect to the servlet, download the task and
    // will be calling functions from EvEvolutionInterface to let know
    // what's happening in the evolution
    interaction.start();

  }

  /* REPORTING PART OF THE APPLICATION */
  private int iteration = 0;

  private EvPopulation<?> population = null;

  private int total_time = 0;


  // report it only when it's neccesary
  // 0 - no reporting
  // 1 - most important reports
  // 2 - average reporting
  // 3 - max. reporting
  private void reportln(String information, int importance) {
    if (reporting >= importance) {
      System.out.println(information);
    }
  }


  /**
   * {@inheritDoc}
   */
  public void addEvalTime(int time) {
    total_time += time;
  }


  /**
   * {@inheritDoc}
   */
  public void addExportedPopulation(List population) {
    reportln("Exported population of " + population.size()
        + " individuals to server", 3);
  }


  /**
   * {@inheritDoc}
   */
  public void addImportedPopulation(List pop) {
    String best = "-";
    EvPopulation population = new EvPopulation(pop);

    if (population.size() != 0)
      best = population.getBestResult().getObjectiveFunctionValue() + "";
    reportln("Imported population of " + population.size() + " ind. Best: ("
        + best + ") " + population.getBestResult(), 2);
  }


  /**
   * {@inheritDoc}
   */
  public void addOperatortime(EvOperator operator, int time) {
  }


  /**
   * {@inheritDoc}
   */
  public void currentCellID(long cell_id) {
    reportln("Current cell id: " + cell_id, 2);

  }


  /**
   * {@inheritDoc}
   */
  public void currentNode(int node_id) {
    reportln("Current node id: " + node_id, 2);
  }


  /**
   * {@inheritDoc}
   */
  public void currentPopulation(EvPopulation population) {
    this.population = population;

  }


  /**
   * {@inheritDoc}
   */
  public void currentTask(int task_id) {
    reportln("Task id: " + task_id, 1);
  }


  /**
   * {@inheritDoc}
   */
  public void iterationProgress() {
    iteration++;
    String best = "";
    if (population != null && population.size() != 0) {
      best =
          "Best (" + population.getBestResult().getObjectiveFunctionValue()
              + "): " + population.getBestResult();
    }
    reportln("Current iteration: " + iteration + " " + best, 2);
  }


  /**
   * {@inheritDoc}
   */
  public void newTask() {
    iteration = 0;
    if (population != null) {
      reportln("Task finished, best individual: ("
          + population.getBestResult().getObjectiveFunctionValue() + ") "
          + population.getBestResult().toString(), 1);
      // serializing best individual:
      if (output_individual_file != null) {
        // TODO
      }
    }
    reportln("New task started!", 1);
  }


  /**
   * {@inheritDoc}
   */
  public void populationSize(int size) {
    reportln("Population size: " + size, 3);
  }

}
