package pl.wroc.uni.ii.evolution.servlets.databasesupport.communication;

import java.io.IOException;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvIndividualInfo;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

public interface EvDBServletCommunication {

  /** ***************************************************************** */
  /** Uploading individuals * */

  public abstract int addIndividual(Object individual, long task_id,
      double objectiveValue, long cell_id, long node_id) throws IOException;


  public abstract void addIndividuals(Object[] individuals, long task_id,
      double[] objective_values, long cell_id, long node_id) throws IOException;


  /** ***************************************************************** */
  /** Deleting individuals * */

  public abstract boolean deleteIndividual(int id) throws IOException;


  public abstract void deleteIndividualsFromTask(long task_id)
      throws IOException;


  /** ***************************************************************** */
  /** Downloading individual * */

  public abstract EvIndividualInfo getIndividualInfo(int ID,
      boolean with_individual_itself) throws IOException;


  public abstract EvIndividualInfo[] getBestIndividualInfos(long task_id,
      int k, int n, boolean with_individuals_itselves) throws IOException;


  public abstract EvIndividualInfo[] getBestIndividualInfosMatchingCell(
      long task_id, long creationCell, int k, int n,
      boolean with_individuals_itselves) throws IOException;


  public abstract EvIndividualInfo[] getBestIndividualInfosNotMatchingCell(
      long task_id, long creationCell, int k, int n,
      boolean with_individuals_itselves) throws IOException;


  public abstract EvIndividualInfo[] getBestIndividualInfosMatchingNode(
      long task_id, long node_id, int k, int n,
      boolean with_individuals_itselves) throws IOException;


  /** *************************************************************************** */
  /** Individual count * */

  public abstract int getIndividualCount(long task_id) throws IOException;


  public abstract int getIndividualCountMatchingCell(long task_id, long cell_id)
      throws IOException;


  public abstract int getIndividualCountNotMatchingCell(long task_id,
      long cell_id) throws IOException;


  public abstract int getIndividualCountMatchingNode(long task_id, long node_id)
      throws IOException;


  /** ***************************************************************** */
  /** SOLUTION SPACE * */

  public abstract Object getSolutionSpace(long task_id, long cell_id)
      throws IOException;


  public abstract int addSolutionSpace(long task_id, long cell_id, Object space)
      throws IOException;


  public abstract boolean deleteSolutionSpaces(long task_id) throws IOException;


  public abstract int getVersionOfNewSolutonSpace(long task_id, long cell_id)
      throws IOException;


  /** ***************************************************************** */
  /** TASKS * */

  public abstract Long[] getTaskIDs() throws IOException;


  public abstract EvTaskInfo getTaskForSystem(int id) throws IOException;


  public abstract int addTaskForSystem(byte[] file, String desc)
      throws IOException;


  public abstract void deleteTaskFromSystem(int id) throws IOException;


  public abstract Integer[] getTaskIDsForSystem() throws IOException;


  public abstract void changeTaskState(int id, int state) throws IOException;


  /** ***************************************************************** */
  /** Resources * */
  public abstract void setResource(Object res, String name) throws IOException;


  public abstract Object getResource(String name) throws IOException;


  public abstract String[] getResourceNames() throws IOException;


  public abstract void deleteResource(String name) throws IOException;


  /** ***************************************************************** */
  /** STATISTICS * */
  public EvStatistic[] getStatistics(long task_id, long cell_id, long node_id)
      throws IOException;


  public EvStatistic[] getStatisticsByTask(long task_id) throws IOException;


  public EvStatistic[] getStatisticsByNodes(long task_id, int[] nodes)
      throws IOException;


  public EvStatistic[] getStatisticsByCells(long task_id, int[] cells)
      throws IOException;


  public void saveStatistic(long task_id, long cell_id, long node_id,
      Object stat, int iteration) throws IOException;


  public Long[] getTaskIdsWithStatistics() throws IOException;


  public Long[] getCellIdsWithStatistics(long task_id) throws IOException;


  public Long[] getNodesIdsWithStatistics(long task_id, long node_id)
      throws IOException;


  public void deleteStatisticForTask(long task_id) throws IOException;


  /** ****************************************************************** */
  /** Eval upload * */

  public int[] addIndividualsToEval(int task_id, long cell_id, long node_id,
      int iter, Object[] individuals) throws Exception;


  public Object[] getIndividualsToEval(int[] ids) throws Exception;


  public void addIndividualsValues(int[] ids, double values[]) throws Exception;


  public double[] getValues(int[] ids) throws Exception;


  public void deleteIndividualsToEval(int task_id) throws Exception;


  public Object[] getIndividualsToEvalByIteration(int task_id, long cell_id,
      long node_id, int iteration) throws Exception;


  /** ===================================================================* */
  public boolean presentFun(int task_id) throws Exception;


  public void addFun(int task_id, Object fun) throws Exception;


  public Object getFun(int task_id) throws Exception;


  public void deleteFun(int task_id) throws Exception;
}