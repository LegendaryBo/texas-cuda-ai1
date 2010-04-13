package pl.wroc.uni.ii.evolution.distribution.workers;

/**
 * Class used for execute EvolutionaryTask in seperate thread and controling it
 * execution
 * 
 * @author Marcin Golebiowski
 */
public class EvTaskThread implements Runnable {

  private Runnable task;

  private Thread worker;


  public EvTaskThread(Runnable task) {
    this.task = task;
  }


  /**
   * Start EvolutionaryTask in seperate thread
   */
  public void start() {
    worker = new Thread(this);
    worker.start();
  }


  /**
   * Stop execution of EvolutionaryTask
   */
  public void stop() {
    EvConsole.println("EvolutionaryTask stopping...");
    worker.interrupt();
    try {
      worker.join();
      EvConsole.println("EvolutionaryTask stoped");
    } catch (InterruptedException e) {
      EvConsole
          .println("Error: Joining interrupted thread for EvolutionaryTask failed");
      return;
    }
  }


  public void run() {
    if (task != null) {
      task.run();
    }
  }


  public boolean isRunning() {
    return worker.isAlive();
  }
}
