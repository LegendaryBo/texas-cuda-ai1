package pl.wroc.uni.ii.evolution.distribution.workers;

public interface EvJARCache {

  /** Returns URL to local JAR file for given node and task * */
  public abstract String getJARUrl(long node_id, long task_id);


  public abstract void clean();

}