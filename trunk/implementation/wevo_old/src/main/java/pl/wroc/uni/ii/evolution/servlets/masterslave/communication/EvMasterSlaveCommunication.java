package pl.wroc.uni.ii.evolution.servlets.masterslave.communication;

import pl.wroc.uni.ii.evolution.servlets.masterslave.EvWorkInfo;

public interface EvMasterSlaveCommunication {

  public long addWork(int task_id, int[] ids) throws Exception;


  public boolean deleteWork(int task_id, long work_id) throws Exception;


  public boolean isWorkDone(long work_id) throws Exception;


  public boolean registerComputaionNode(int task_id, long comp_node_id,
      double benchmark) throws Exception;


  public boolean unregisterComputationNode(int task_id, long comp_node_id)
      throws Exception;


  public EvWorkInfo getWork(int task_id, long comp_node_id) throws Exception;


  public boolean informWorkDone(EvWorkInfo work) throws Exception;


  public boolean keepalive(long comp_node_id) throws Exception;


  public long[] getWorks() throws Exception;


  public long[] getNodesForTask(int task) throws Exception;

}
