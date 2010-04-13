package pl.wroc.uni.ii.evolution.servlets.databasesupport.providers;

import java.io.IOException;
import java.sql.Connection;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvIndividualInfo;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

public interface EvDatabaseProvider {

  public void init();


  /**
   * Stores <code> individual </code> in database
   * 
   * @param individual
   * @param task_id
   * @param objective_value
   * @param cell_id
   * @param node_id
   * @return
   * @throws IOException
   */
  public int addIndividual(Connection connection, Object individual,
      long task_id, double objective_value, long cell_id, long node_id)
      throws IOException;


  /**
   * Stores <code> individual </code> in database
   * 
   * @param individual
   * @param task_id
   * @param objective_value
   * @param cell_id
   * @param node_id
   * @return
   * @throws IOException
   */
  public int[] addIndividuals(Connection connection, Object[] individuals,
      long task_id, double[] objective_values, long cell_id, long node_id)
      throws IOException;


  /**
   * Delete individual having given <code> id </code>
   * 
   * @param id
   * @throws IOException
   */
  public boolean deleteIndividual(Connection connection, int id)
      throws IOException;


  /**
   * Delete all individuals from given task
   * 
   * @param task_id
   * @throws IOException
   */
  public void deleteIndividualsFromTask(Connection connection, long task_id)
      throws IOException;


  /**
   * Gets id's of all task in database
   */
  public Long[] getTaskIDs(Connection connection) throws IOException;


  /**
   * Return solution space for given cell and task
   * 
   * @param task_id
   * @param cell_id
   * @return
   * @throws IOException
   */
  public Object getSolutionSpace(Connection connection, long task_id,
      long cell_id) throws IOException;


  /**
   * Add a solution space for given task and cell to database
   * 
   * @param space
   * @param task_id
   * @param cell_id
   * @throws IOExeption
   */
  public int addSolutionSpace(Connection connection, long task_id,
      long cell_id, Object space) throws IOException;


  /**
   * Deletes solution spaces for given task
   * 
   * @param task_id
   * @throws IOException
   */
  public boolean deleteSolutionSpaces(Connection connection, long task_id)
      throws IOException;


  /**
   * Return information about individuals having given <code> id </code>
   * 
   * @param id
   * @param with_individual_itself if set <code> true </code> then individuals
   *        is also returned
   * @return
   * @throws IOException
   */
  public EvIndividualInfo getIndividualInfo(Connection connection, int id,
      boolean with_individual_itself) throws IOException;


  /**
   * Returns description of n best individuals starting from k-th. In other
   * words, returns descriptions of I_k, I_{k+1}, ..., I_{k+n-1}. The best
   * individual in the database is I_1. If there are no (k+n-1) individuals in
   * the database, the end part of the table returned is filled by nulls.
   */
  public EvIndividualInfo[] getBestIndividualInfos(Connection connection,
      long task_id, int k, int n, boolean with_individuals_itselves)
      throws IOException;


  /**
   * The same as getBestIndividualInfos, but applied only to individuals with a
   * specific creation cell.
   */
  public EvIndividualInfo[] getBestIndividualInfosMatchingCell(
      Connection connection, long task_id, long cell_id, int k, int n,
      boolean with_individuals_itselves) throws IOException;


  /**
   * The same as getBestIndividualInfos, but applied only to individuals with a
   * creation cell different from the specific one.
   */
  public EvIndividualInfo[] getBestIndividualInfosNotMatchingCell(
      Connection connection, long task_id, long cell_id, int k, int n,
      boolean with_individuals_itselves) throws IOException;


  /**
   * The same as getBestIndividualInfos, but applied only to individuals with a
   * specific node.
   */
  public EvIndividualInfo[] getBestIndividualInfosMatchingNode(
      Connection connection, long task_id, long node_id, int k, int n,
      boolean with_individuals_itselves) throws IOException;


  /**
   * Retruns how many individuals are stored in a database for given task
   * 
   * @param task_id
   * @return
   * @throws IOException
   */
  public int getIndividualCount(Connection connection, long task_id)
      throws IOException;


  /**
   * Returns how many individuals are stored in a database for given task,
   * created by nodes in given cell
   * 
   * @param task_id
   * @param cell_id
   * @return
   * @throws IOException
   */
  public int getIndividualCountMatchingCell(Connection connection,
      long task_id, long cell_id) throws IOException;


  /**
   * Returns how many individuals are stored in database for given task, created
   * by cells different than given
   * 
   * @param task_id
   * @param cell_id
   * @return
   * @throws IOException
   */
  public int getIndividualCountNotMatchingCell(Connection connection,
      long task_id, long cell_id) throws IOException;


  /**
   * Returns the highest interation number of solution space stored in database
   * for given task and cell
   * 
   * @param task_id
   * @param cell_id
   * @return
   * @throws IOException
   */
  public int getVersionOfNewSolutonSpace(Connection connection, long task_id,
      long cell_id) throws IOException;


  /**
   * Returns how many individuals are stored in database for given task, created
   * by given node
   * 
   * @param task_id
   * @param node_id
   * @return
   * @throws IOException
   */
  public int getIndividualCountMatchingNode(Connection connection,
      long task_id, long node_id) throws IOException;


  /** ***************************************************************** */
  /** TASKS * */
  public int addTaskForSystem(Connection connection, byte[] file, String desc)
      throws IOException;


  public EvTaskInfo getTaskForSystem(Connection connection, int id)
      throws IOException;


  public void deleteTaskFromSystem(Connection connection, int id)
      throws IOException;


  public Integer[] getTaskIDsForSystem(Connection connection)
      throws IOException;


  public void changeTaskState(Connection connection, int id, int state)
      throws IOException;


  /** ***************************************************************** */
  /** Resources * */
  public boolean setResource(Connection connection, Object res, String name)
      throws IOException;


  public Object getResource(Connection connection, String name)
      throws IOException;


  public String[] getResourceNames(Connection connection) throws IOException;


  public boolean deleteResource(Connection connection, String name)
      throws IOException;


  /** ***************************************************************** */
  /** STATISTICS * */

  public Object[] getStatistics(Connection connection, long task_id,
      long cell_id, long node_id) throws IOException;


  public Object[] getStatisticsByCells(Connection connection, long taks_id,
      int[] cells);


  public Object[] getStatisticsByNodes(Connection connection, long taks_id,
      int[] nodes);


  public Object[] getStatisticsByID(Connection connection, long task_id);


  public void saveStatistic(Connection connection, long task_id, long cell_id,
      long node_id, Object stat, long iteration) throws IOException;


  public Long[] getTaskIdsWithStatistics(Connection connection)
      throws IOException;


  public Long[] getCellIdsWithStatistics(Connection connection, long task_id)
      throws IOException;


  public Long[] getNodesIdsWithStatistics(Connection connection, long task_id,
      long cell_id) throws IOException;


  public void deleteStatisticForTask(Connection connection, long task_id)
      throws IOException;


  /** ===================================================================== */
  /** EVAL * */

  public int[] addIndividualsToEval(Connection connection, int task_id,
      long cell_id, long node_id, int iteration, Object[] individuals)
      throws Exception;


  public Object[] getIndividualsToEval(Connection connection, int[] ids);


  public void addIndividualsValues(Connection connection, int[] ids,
      double values[]);


  public double[] getValues(Connection connection, int[] ids);


  public void deleteEvaledIndividuals(Connection connection, int task_id);


  public Object[] getIndividualsToEvalByIter(Connection conn, int task_id,
      long cell_id, long node_id, int iteration);


  /** =================================================================== */
  /** FUN * */
  public boolean presentFun(Connection connection, int task_id);


  public void addFun(Connection connection, int task_id, Object fun);


  public Object getFun(Connection connection, int task_id);


  public void deleteFun(Connection conn, int task_id);
}