package pl.wroc.uni.ii.evolution.distribution.workers;

/**
 * It loads EvolutionaryTask from JAR. Manifest of this JAR must have attribute:
 * IslandCreator. Value of this attribute must be a name of class implementing
 * IslandCreator interface.
 * 
 * @author Piotr Lipinski, Marcin Golebiowski, Kamil Dworakowski
 */
public interface EvTaskLoader {

  /**
   * Returns instance of EvIsland that is created by some class in JAR file.
   * Name of this class is stored in JAR's manifest.
   * 
   * @param jar_url
   * @param task_id
   * @param node_id
   * @param type 0 - evolution task, 1 - obj. function evaluation
   * @return EvolutionaryTask
   */
  public Runnable getTask(String jar_url, int task_id, long node_id, int type,
      EvEvalTaskInterface inter, EvEvolutionInterface evol_inter);
}