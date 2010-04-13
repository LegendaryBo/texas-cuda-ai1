package pl.wroc.uni.ii.evolution.distribution.workers.console.Eval;

import java.util.Date;

import pl.wroc.uni.ii.evolution.distribution.workers.EvEvalTaskInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvJARCacheImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoader;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoaderImpl;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskMaster;
import pl.wroc.uni.ii.evolution.distribution.workers.console.EvConsoleParameters;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunicationImpl;

/**
 * Application class implementing program which communicate with wEvo framework
 * and evaluates objective functions sent from there.<br>
 * This process is repeated till the program is closed.<br>
 * Application implements EvEvalTaskInterface, it's function are invoked by
 * evaluation task which is run in application.<br>
 * <br>
 * <br>
 * Parameters:<br>
 * -s server - connect to specified URL. If parameter is not given, the
 * application uses address http://127.0.0.1:8080/<br>
 * <br>
 * Examples:<br>
 * java EvConsoleEval -s http://192.168.0.1:8080/
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvConsoleEval implements EvEvalTaskInterface {

  private long work_id;

  private int previous_download_time = 0;

  private int previous_upload_time = 0;

  private int previous_eval_time = 0;


  public static void main(String[] args) {

    EvConsoleParameters parameters = new EvConsoleParameters(args);
    EvConsoleEval this_application = new EvConsoleEval();

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
    jar_manager.init(System.currentTimeMillis());

    EvTaskMaster interaction =
        new EvTaskMaster(wevo_manager_server, loader, jar_manager, 2000, 1,
            this_application);

    /** start interaction with management servlet */
    // application will connect to the servlet, download the task and
    // will be calling functions from EvEvalTaskInterface to let know
    // what's happening in the evolution
    interaction.start();

  }


  /**
   * {@inheritDoc}
   */
  public void addDownloadTime(int miliseconds) {
    previous_download_time = miliseconds;
  }


  /**
   * {@inheritDoc}
   */
  public void addEvaluationTime(int miliseconds) {
    previous_eval_time = miliseconds;
  }


  /**
   * {@inheritDoc}
   */
  public void addUploadTime(int miliseconds) {
    previous_upload_time = miliseconds;
  }


  /**
   * {@inheritDoc}
   */
  public void addwaitingTime(int miliseconds) {
  }


  /**
   * {@inheritDoc}
   */
  public void currentCellID(long cell_id) {
    reportln("cell id: " + cell_id);
  }


  /**
   * {@inheritDoc}
   */
  public void currentNodeID(long node_id) {
    reportln("node id: " + node_id);

  }


  /**
   * {@inheritDoc}
   */
  public void currentObjectiveFunction(String objFun) {
    reportln("Evaluated objective function: " + objFun);

  }


  /**
   * {@inheritDoc}
   */
  public void currentState(int state) {
    if (state == 1) {
      reportln("Waiting for job...");
      reportln("Download time: " + previous_download_time + "ms "
          + "Upload time: " + previous_upload_time + "ms Evaluation time: "
          + previous_eval_time + "ms");

    }
    if (state == 2) {
      reportln("Started evaluating individuals...");
    }
    if (state == 3) {
      reportln("Performing benchmark for server...");
    }
    if (state == 4) {
      reportln("Downloading new individuals from server...");
    }
    if (state == 5) {
      reportln("Uploading results to server...");
    }
  }


  /**
   * {@inheritDoc}
   */
  public void currentTaskID(int task_id) {
    reportln("Current work id:" + task_id);

  }


  /**
   * {@inheritDoc}
   */
  public void currentWorkID(long work_id) {
    this.work_id = work_id;
    System.out.println("");
    reportln("Current work id:" + work_id);
  }


  /**
   * {@inheritDoc}
   */
  public void individualEvaluated() {

  }


  /**
   * {@inheritDoc}
   */
  public void jobAborted() {
    reportln("Job " + work_id + " has been aborted by server");

  }


  /**
   * {@inheritDoc}
   */
  public void jobFinished() {
    reportln("Job " + work_id + " finished");
  }


  /**
   * {@inheritDoc}
   */
  public void newJobSize(int length) {
    reportln("New job size:" + length);
  }


  private void reportln(String information) {

    Date date_obj = new Date();

    String date =
        date_obj.getHours() + ":" + date_obj.getMinutes() + ":"
            + date_obj.getSeconds();

    System.out.println(date + " " + information);

  }

}
