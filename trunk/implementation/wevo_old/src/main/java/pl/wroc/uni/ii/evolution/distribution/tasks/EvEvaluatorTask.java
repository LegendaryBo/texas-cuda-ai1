package pl.wroc.uni.ii.evolution.distribution.tasks;

public class EvEvaluatorTask implements EvTaskCreator {

  public Runnable create(int task_id, long node_id, String wevo_server_url) {
    return new EvEvalutor(wevo_server_url, node_id, task_id);
  }

}
