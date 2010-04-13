package pl.wroc.uni.ii.evolution.servlets.masterslave;

public class EvWorkInfo {

  public int task_id;

  public long work_id;

  public int[] ids;


  public EvWorkInfo(int task_id, long work_id, int[] ids) {
    this.task_id = task_id;
    this.work_id = work_id;
    this.ids = ids;
  }
}
