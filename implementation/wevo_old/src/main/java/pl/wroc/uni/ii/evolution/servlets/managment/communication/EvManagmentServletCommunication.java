package pl.wroc.uni.ii.evolution.servlets.managment.communication;

import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

/**
 * Interface to all classes used as proxy to managment servlet
 * 
 * @author Marcin Golebiowski, Kamil Dworakowski
 */
public interface EvManagmentServletCommunication {

  /**
   * Registers node on managment servet and returns node identifier
   * 
   * @return long
   * @throws Exception
   */
  long getNodeID() throws Exception;


  /**
   * Retruns task identifier
   * 
   * @param node_id
   * @return int
   * @throws Exception
   */
  int getTaskID(long node_id) throws Exception;


  /**
   * Return if continue current computation.
   * 
   * @param node_id
   * @return boolean
   * @throws Exception
   */
  boolean keepAlive(long node_id) throws Exception;


  /**
   * Stops execution of task in system
   * 
   * @param task_id
   * @return true if succeed, false if not
   */
  void stopTask(int task_id) throws Exception;


  /**
   * Resumes execution of task in system
   * 
   * @param task_ik
   */
  void resumeTask(int task_id) throws Exception;


  /**
   * Deletes task from system
   * 
   * @param task_id
   */
  void deleteTask(int task_id) throws Exception;


  /**
   * Sends file with for system to management servlet
   * 
   * @param file
   * @param desc
   * @return
   * @throws Exception
   */
  int addTask(byte file[], String desc) throws Exception;


  /**
   * Fetchs EvTask from managment servlet
   */
  EvTaskInfo getEvTask(int task_id, boolean with_jar) throws Exception;


  /**
   * Returns list of task identifiers
   */
  int[] getEvTaskIds() throws Exception;


  /** Gets JAR for node */

  byte[] getJAR(long node_id) throws Exception;


  /**
   * Returns how many nodes are working on given task
   * 
   * @param task_id
   * @return
   * @throws Exception
   */
  int getNodeCountForTask(int task_id) throws Exception;


  /**
   * Delete all individuals computed for the task
   * 
   * @param task_id
   * @throws Exception
   */
  void clearTask(int task_id) throws Exception;


  /**
   * Returns a stored url to management servlet
   */
  String getURL();

}
