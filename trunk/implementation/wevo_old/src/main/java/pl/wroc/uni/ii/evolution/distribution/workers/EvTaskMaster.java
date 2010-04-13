package pl.wroc.uni.ii.evolution.distribution.workers;

import java.io.IOException;

import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;

/**
 * The highest level manager in the applet. It
 * <li> connects to the server to obtain client_id, task_id
 * <li> connects to the server to obtain keepAliveOK or keepAliveSuice
 * <li> downloads jar with a task and executes it
 * <li> stays in touch with a server and interrupts task in case of order from
 * the server
 * 
 * @author Marcin Golebiowski, Kamil Dworakowski
 */
public class EvTaskMaster implements Runnable {

  private EvManagmentServletCommunication proxy;

  private EvTaskLoader loader;

  private EvJARCache jar_cache;

  private Thread task_master_thread;

  private EvTaskThread task_thread;

  private int delay;

  private long node_id;

  private int task_id;

  // task type 0-evolution task, 1-obj. function evaluation task
  private int type;

  private EvEvalTaskInterface inter;

  private EvEvolutionInterface evol_inter;


  public EvTaskMaster(EvManagmentServletCommunication proxy,
      EvTaskLoader loader, EvJARCache jar_manager, int delay, int type,
      EvEvalTaskInterface inter) {
    this.inter = inter;
    this.jar_cache = jar_manager;
    this.proxy = proxy;
    this.loader = loader;
    this.delay = delay;
    this.type = type;
  }


  public EvTaskMaster(EvManagmentServletCommunication proxy,
      EvTaskLoader loader, EvJARCache jar_manager, int delay, int type,
      EvEvolutionInterface evol_inter) {
    this.evol_inter = evol_inter;
    this.jar_cache = jar_manager;
    this.proxy = proxy;
    this.loader = loader;
    this.delay = delay;
    this.type = type;
  }


  /**
   * Interaction with managment servlet
   */
  public void run() {

    while (true) {
      try {
        Thread.sleep(delay);
        // EvConsole.println("Connecting to: " + proxy.getURL());

        /** get clientID */
        node_id = proxy.getNodeID();
        // EvConsole.println("Node id = " + node_id);

        /** get TASK */
        Runnable task = getTask();
        if (task == null) {
          EvConsole.println("Problem with fetching task");
          EvConsole.println("Cleaning cache");
          jar_cache.clean();
          Thread.sleep(delay);
          continue;
        }

        /** START TASK */
        // EvConsole.println("Start executing JAR with wEvo task");
        task_thread = new EvTaskThread(task);
        task_thread.start();

        /** CONTROL EXECUTION OF TASK */
        stayInTouch(task);

      } catch (InterruptedException ex) {

        EvConsole.println("TaskMaster.run() interrupted");
        EvConsole.println("Stopping EvolutionaryTaskThread...");

        if (task_thread != null) {
          task_thread.stop();
        }
        return;
      } catch (IOException ex) {
        EvConsole.println("Problem occurred: (connection error) ");
        delay += 1000;
      } catch (Exception ex) {
        EvConsole.println("Problem occurred: (" + ex.getMessage() + ")");
        delay += 1000;
      }

    }
  }


  private void stayInTouch(Runnable task) throws InterruptedException,
      Exception {
    for (;;) {
      Thread.sleep(delay);
      boolean continue_task = proxy.keepAlive(node_id);

      if (!continue_task) {
        EvConsole.println("Task was interupted due to server request");
        EvConsole.println("not continue task ");
        System.err.println("Task interupted");

        task_thread.stop();
        break;
      }

      if (!task_thread.isRunning()) {
        EvConsole.println("Task " + task_id + ": end");

        break;
      }
    }
  }


  /**
   * Starts interaction with managment servlet
   */
  public void start() {
    task_master_thread = new Thread(this);
    task_master_thread.start();
  }


  /**
   * Stops interaction with managment servlet
   */
  public void stop() {
    if (task_master_thread == null) {
      return;
    }
    task_master_thread.interrupt();
    try {
      task_master_thread.join();
    } catch (InterruptedException e) {
      EvConsole
          .println("Error: join interrupted Thread running TaskMaster failed");
    }
  }


  private Runnable getTask() throws Exception {
    /** get TaskID */
    task_id = proxy.getTaskID(node_id);
    // EvConsole.println("Task id = " + task_id);
    // EvConsole.println("Fetching JAR with task");

    /** get URL to JAR file with task */
    String jar_url = jar_cache.getJARUrl(node_id, task_id);
    /** load task */
    Runnable task =
        loader.getTask(jar_url, task_id, node_id, type, inter, evol_inter);
    // EvConsole.println("bla"+task);

    return task;
  }
}
