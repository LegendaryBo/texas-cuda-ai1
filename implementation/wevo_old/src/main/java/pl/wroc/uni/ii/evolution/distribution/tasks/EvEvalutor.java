package pl.wroc.uni.ii.evolution.distribution.tasks;

import pl.wroc.uni.ii.evolution.distribution.workers.EvEvalTaskInterface;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.servlets.masterslave.EvWorkInfo;
import pl.wroc.uni.ii.evolution.servlets.masterslave.communication.EvMasterSlaveCommunication;
import pl.wroc.uni.ii.evolution.servlets.masterslave.communication.EvMasterSlaveCommunicationImpl;
import pl.wroc.uni.ii.evolution.utils.benchmark.EvBenchmark;

/**
 * Implementation of thread which evaluates objective function values of
 * received individuals.<BR>
 * It's run inside aplet or console. It communicates with it's caller using
 * EvEvalTaskInterface
 * 
 * @author Marcin Golebiowski, Kacper Gorski
 */
public class EvEvalutor implements Runnable {

  private EvMasterSlaveCommunication ev_comm;

  private EvDBServletCommunicationImpl db_comm;

  private long node_id;

  private int task_id;

  private EvEvalTaskInterface task_interface = null;

  private Thread keepalive;

  private Thread synchronizer;

  public boolean interrupt = false; // check this to true to interrupt
                                    // evaluation of obj_fun

  public long currwork = -1;


  /**
   * Default contructor. Run method starts thread
   * 
   * @param wevo_server_urll
   * @param node_id - node_id assigned to this pc
   * @param task_id - task_id assigned to this pc
   */
  public EvEvalutor(String wevo_server_url, long node_id, int task_id) {

    this.node_id = node_id;
    this.task_id = task_id;

    this.ev_comm = new EvMasterSlaveCommunicationImpl(wevo_server_url);
    this.db_comm = new EvDBServletCommunicationImpl(wevo_server_url);
  }


  public void setInterface(EvEvalTaskInterface task_interface) {
    this.task_interface = task_interface;
  }


  @SuppressWarnings("unchecked")
  public void run() {

    // System.out.println("Start benchmark");
    if (task_interface == null)
      throw new IllegalStateException(
          "Interface beetwen task and "
              + "it's master is not set! (Call setInterface method before running task)");
    task_interface.currentState(3);
    double benchmark = EvBenchmark.runMasterSlaveBenchmark();
    // System.out.println("End benchmark");

    try {
      // System.out.println("Register computation node");
      ev_comm.registerComputaionNode(task_id, node_id, benchmark);
      // System.out.println("TaskId =" + task_id);
      task_interface.currentTaskID(task_id);
      // System.out.println("NodeID = " + node_id);
      // System.out.println("Register node ok");
    } catch (Exception e1) {
      e1.printStackTrace(System.out);
    }
    task_interface.currentState(1);
    // System.out.println("Start keepalive");

    this.keepalive = new Thread(new Keepalive(ev_comm, node_id));
    this.keepalive.start();
    this.synchronizer = new Thread(new Synchronizer(ev_comm, this));
    this.synchronizer.start();

    while (true) {
      interrupt = false;
      try {
        // get indexes to evaluate from master slave servlet
        EvWorkInfo work = ev_comm.getWork(task_id, node_id);

        if (work == null) {
          Thread.sleep(1000);
          task_interface.addwaitingTime(1000);
          currwork = -1;
          continue;
        } else
          currwork = work.work_id;
        task_interface.currentState(4);
        int[] ids = work.ids;
        task_interface.currentWorkID(work.work_id);
        task_interface.currentNodeID(node_id);
        // System.out.println("------");
        // System.out.println("Work ID = " + work.work_id);
        // get those individuals
        // System.out.println("Number = " + ids.length);
        task_interface.newJobSize(ids.length);
        if (ids.length != 0) {

          long start_d = System.currentTimeMillis();
          Object[] individuals = db_comm.getIndividualsToEval(ids);
          long end_d = System.currentTimeMillis();
          task_interface.addDownloadTime((int) (end_d - start_d));

          // System.out.println("Download = " + (end_d - start_d) / 1000 + "
          // s");

          EvObjectiveFunction<EvIndividual> function =
              (EvObjectiveFunction) db_comm.getFun(task_id);
          task_interface.currentObjectiveFunction(function.toString());
          // System.out.println("Objective function downloaded");
          // System.out.println("Start");
          // evaluating objective function
          double[] values = new double[individuals.length];
          task_interface.currentState(2);
          long start_ev = System.currentTimeMillis();
          boolean go_back = false;
          for (int i = 0; i < individuals.length; i++) {
            EvIndividual individual = (EvIndividual) individuals[i];
            individual.setObjectiveFunction(function); // setting objective
                                                        // function
            task_interface.individualEvaluated();
            if (interrupt == true) {
              interrupt = false;
              // System.out.println("Evaluation interrupted - job was done by
              // another PC");
              go_back = true;
              break;
            }
            values[i] = individual.getObjectiveFunctionValue();
          }
          if (go_back) {
            task_interface.jobAborted();
            long end_ev = System.currentTimeMillis();
            task_interface.addEvaluationTime((int) (end_ev - start_ev));
            task_interface.currentState(1);
            continue;
          }
          long end_ev = System.currentTimeMillis();
          task_interface.addEvaluationTime((int) (end_ev - start_ev));
          // System.out.println("EVAL = " + (end_ev - start_ev) / 1000 + " s");
          // send evaluated values
          task_interface.currentState(5);
          long start_s = System.currentTimeMillis();
          db_comm.addIndividualsValues(ids, values);
          long end_s = System.currentTimeMillis();
          task_interface.addUploadTime((int) (end_s - start_s));
          // System.out.println("SENDING VALUES = " + (end_s - start_s)/1000 +"
          // s");
        }
        // inform eval servlet it's done
        ev_comm.informWorkDone(work);
        task_interface.currentState(1);
      } catch (InterruptedException e) {
        return;
      } catch (Exception e) {
        e.printStackTrace(System.out);
      }
    }
  }

}

class Keepalive implements Runnable {

  private long node_id;

  private EvMasterSlaveCommunication ev_comm;


  public Keepalive(EvMasterSlaveCommunication ev_comm, long node_id) {
    this.node_id = node_id;
    this.ev_comm = ev_comm;
  }


  public void run() {

    for (;;) {

      try {
        ev_comm.keepalive(node_id);
        Thread.sleep(10000);
      } catch (Exception e) {
        return;
      }
    }
  }
}

// this thread connects every specified number of miliseconds
// to evalServlet to chek if current work is done by another computer
class Synchronizer implements Runnable {
  private final int delay = 333;

  private EvMasterSlaveCommunication ev_comm;

  private EvEvalutor evalutor;


  public Synchronizer(EvMasterSlaveCommunication ev_comm, EvEvalutor evalutor) {
    this.ev_comm = ev_comm;
    this.evalutor = evalutor;
  }


  public void run() {
    while (true) {

      try {
        Thread.sleep(delay);
        if (evalutor.currwork != -1 && ev_comm.isWorkDone(evalutor.currwork)) {
          evalutor.interrupt = true;
        }

      } catch (Exception e) {
        e.printStackTrace();
      }

    }
  }

}