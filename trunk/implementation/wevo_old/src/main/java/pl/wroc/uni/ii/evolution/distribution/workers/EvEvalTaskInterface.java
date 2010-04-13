package pl.wroc.uni.ii.evolution.distribution.workers;

// TODO implement console program using this interface, so that wEvo can
// be used on computers without browser

/**
 * @author Kacper Gorski This interface is a connection beetwen wEvo aplet (or
 *         console) and ovjective function evaluation task. It contains
 *         functions that let know implementing object what is happening inside
 *         task. Currently used in EvEvalApplet
 */
public interface EvEvalTaskInterface {

  // inform interface that one of the individuals has been evaluated
  public void individualEvaluated();


  // let know interface about time spend on evaluation
  public void addEvaluationTime(int miliseconds);


  // let know interface about wasted time that wasn't spend on evaluation
  public void addwaitingTime(int miliseconds);


  // single job is finished
  public void jobFinished();


  // id of the task currently evaluated
  public void currentTaskID(int task_id);


  // id of the node currently evaluated
  public void currentNodeID(long node_id);


  // id of the cell currently evaluated
  public void currentCellID(long cell_id);


  // inform about title of objective function currently evaluated
  public void currentObjectiveFunction(String objFun);


  public void addDownloadTime(int miliseconds);


  public void addUploadTime(int miliseconds);


  // inform about current state of the task
  // 1 - waiting for job
  // 2 - evaluating
  // 3 - benchmarking
  // 4 - downloading individuals
  // 5 - uploading individuals
  public void currentState(int state);


  public void newJobSize(int length);


  public void jobAborted();


  public void currentWorkID(long work_id);

}
