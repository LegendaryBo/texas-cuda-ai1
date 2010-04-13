package pl.wroc.uni.ii.evolution.distribution.tasks;

/**
 * Interface run by applet on wEvo framework. To create your evolutionary task
 * overwrite this interface.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @author Kacper Gorski (admin@34all.org)
 */
public interface EvTaskCreator {

  /**
   * @param task_id
   * @param node_id
   * @param wevo_server_url - Url address to your computer with wEvo installed
   *        on it
   * @return program to be run by wevo evolution applet or console.
   */
  public Runnable create(int task_id, long node_id, String wevo_server_url);
}
