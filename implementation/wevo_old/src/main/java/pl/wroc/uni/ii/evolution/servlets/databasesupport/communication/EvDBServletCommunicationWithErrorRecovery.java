package pl.wroc.uni.ii.evolution.servlets.databasesupport.communication;

import java.io.IOException;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvIndividualInfo;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

/**
 * This is a instance of decorator pattern. It adds network error and latency
 * tolerance. In other words it retries an operation several times before
 * reporting problem.
 * 
 * @author Kamil Dworakowski, Marcin Golebiowski
 */
public class EvDBServletCommunicationWithErrorRecovery implements
    EvDBServletCommunication {

  private EvDBServletCommunication actual_gateway;

  int number_of_tries, delay;


  /**
   * @param wevo_server_url
   * @param number_of_tries maximum number of tries after failed attempt to
   *        connect to servlet
   * @param delay miliseconds to wait before retrying a failed attempt
   */
  public EvDBServletCommunicationWithErrorRecovery(String wevo_server_url,
      int number_of_tries, int delay) {

    this.actual_gateway = new EvDBServletCommunicationImpl(wevo_server_url);
    this.number_of_tries = number_of_tries;
    this.delay = delay;

    if (actual_gateway instanceof EvDBServletCommunicationWithErrorRecovery) {
      throw new IllegalArgumentException(
          "actual_gateway for DBGatewayWithErrorRecovery can't be  DBGatewayWithErrorRecovery");
    }
  }


  /**
   * @deprecated
   */
  public EvDBServletCommunicationWithErrorRecovery(
      EvDBServletCommunication gateway, int number_of_tries, int delay) {

    this.actual_gateway = gateway;
    this.number_of_tries = number_of_tries;
    this.delay = delay;

    if (actual_gateway instanceof EvDBServletCommunicationWithErrorRecovery) {
      throw new IllegalArgumentException(
          "actual_gateway for DBGatewayWithErrorRecovery can't be  DBGatewayWithErrorRecovery");
    }
  }


  public int addIndividual(Object individual, long task_id,
      double objective_value, long cell_id, long node_id) throws IOException {

    for (int tries = 0; tries < this.number_of_tries; tries++) {
      try {
        return actual_gateway.addIndividual(individual, task_id,
            objective_value, cell_id, node_id);
      } catch (IOException e) {
      }
      waitAMoment();
    }
    throw new IOException("Problem with connection to database");
  }


  public int addSolutionSpace(long task_id, long cell_id, Object space)
      throws IOException {
    return actual_gateway.addSolutionSpace(task_id, cell_id, space);
  }


  public boolean deleteIndividual(int id) throws IOException {
    return actual_gateway.deleteIndividual(id);
  }


  public void deleteIndividualsFromTask(long task_id) throws IOException {
    actual_gateway.deleteIndividualsFromTask(task_id);
  }


  public boolean deleteSolutionSpaces(long task_id) throws IOException {
    return actual_gateway.deleteSolutionSpaces(task_id);
  }


  public EvIndividualInfo[] getBestIndividualInfos(long task_id, int k, int n,
      boolean with_individuals_itselves) throws IOException {
    return actual_gateway.getBestIndividualInfos(task_id, k, n,
        with_individuals_itselves);
  }


  public EvIndividualInfo[] getBestIndividualInfosMatchingCell(long task_id,
      long cell_id, int k, int n, boolean with_individuals_itselves)
      throws IOException {
    for (int tries = 0; tries < this.number_of_tries; tries++) {
      try {
        return actual_gateway.getBestIndividualInfosMatchingCell(task_id,
            cell_id, k, n, with_individuals_itselves);
      } catch (IOException e) {
      }
      waitAMoment();
    }
    // too many failed attempts
    throw new IOException("Problem with connection to database");
  }


  private void waitAMoment() {
    try {
      Thread.sleep(delay);
    } catch (InterruptedException e) {
      return;
    }
  }


  public EvIndividualInfo[] getBestIndividualInfosMatchingNode(long task_id,
      long node_id, int k, int n, boolean with_individuals_itselves)
      throws IOException {
    return actual_gateway.getBestIndividualInfosMatchingNode(task_id, node_id,
        k, n, with_individuals_itselves);
  }


  public EvIndividualInfo[] getBestIndividualInfosNotMatchingCell(long task_id,
      long cell_id, int k, int n, boolean with_individuals_itselves)
      throws IOException {
    return actual_gateway.getBestIndividualInfosNotMatchingCell(task_id,
        cell_id, k, n, with_individuals_itselves);
  }


  public int getIndividualCount(long task_id) throws IOException {
    return actual_gateway.getIndividualCount(task_id);
  }


  public int getIndividualCountMatchingCell(long task_id, long cell_id)
      throws IOException {
    return actual_gateway.getIndividualCountMatchingCell(task_id, cell_id);
  }


  public int getIndividualCountMatchingNode(long task_id, long node_id)
      throws IOException {
    return actual_gateway.getIndividualCountMatchingNode(task_id, node_id);
  }


  public int getIndividualCountNotMatchingCell(long task_id, long cell_id)
      throws IOException {
    return actual_gateway.getIndividualCountNotMatchingCell(task_id, cell_id);
  }


  public EvIndividualInfo getIndividualInfo(int id,
      boolean with_individual_itself) throws IOException {
    return actual_gateway.getIndividualInfo(id, with_individual_itself);
  }


  public Object getSolutionSpace(long task_id, long cell_id) throws IOException {
    for (int tries = 0; tries < this.number_of_tries; tries++) {
      try {
        return actual_gateway.getSolutionSpace(task_id, cell_id);
      } catch (IOException e) {
      }
      waitAMoment();
    }
    throw new IOException("Problem with connection to database");
  }


  public Long[] getTaskIDs() throws IOException {
    return actual_gateway.getTaskIDs();
  }


  public int getVersionOfNewSolutonSpace(long task_id, long cell_id)
      throws IOException {
    return actual_gateway.getVersionOfNewSolutonSpace(task_id, cell_id);
  }


  public int addTaskForSystem(byte[] file, String desc) throws IOException {
    return actual_gateway.addTaskForSystem(file, desc);
  }


  public void deleteTaskFromSystem(int id) throws IOException {
    actual_gateway.deleteTaskFromSystem(id);
  }


  public EvTaskInfo getTaskForSystem(int id) throws IOException {
    return actual_gateway.getTaskForSystem(id);
  }


  public Integer[] getTaskIDsForSystem() throws IOException {
    return actual_gateway.getTaskIDsForSystem();
  }


  public void changeTaskState(int id, int state) throws IOException {
    actual_gateway.changeTaskState(id, state);
  }


  public void deleteResource(String name) throws IOException {
    actual_gateway.deleteResource(name);
  }


  public Object getResource(String name) throws IOException {
    return actual_gateway.getResource(name);
  }


  public String[] getResourceNames() throws IOException {
    return actual_gateway.getResourceNames();
  }


  public void setResource(Object res, String name) throws IOException {
    actual_gateway.setResource(res, name);
  }


  public void deleteStatisticForTask(long task_id) throws IOException {

    actual_gateway.deleteStatisticForTask(task_id);

  }


  public Long[] getCellIdsWithStatistics(long task_id) throws IOException {
    return actual_gateway.getCellIdsWithStatistics(task_id);
  }


  public Long[] getNodesIdsWithStatistics(long task_id, long node_id)
      throws IOException {
    return actual_gateway.getNodesIdsWithStatistics(task_id, node_id);
  }


  public EvStatistic[] getStatistics(long task_id, long cell_id, long node_id)
      throws IOException {
    return actual_gateway.getStatistics(task_id, cell_id, node_id);
  }


  public Long[] getTaskIdsWithStatistics() throws IOException {
    return actual_gateway.getTaskIdsWithStatistics();
  }


  public void saveStatistic(long task_id, long cell_id, long node_id,
      Object stat, int iteration) throws IOException {
    actual_gateway.saveStatistic(task_id, cell_id, node_id, stat, iteration);
  }


  public EvStatistic[] getStatisticsByCells(long task_id, int[] cells)
      throws IOException {
    return actual_gateway.getStatisticsByCells(task_id, cells);
  }


  public EvStatistic[] getStatisticsByNodes(long task_id, int[] nodes)
      throws IOException {
    return actual_gateway.getStatisticsByNodes(task_id, nodes);
  }


  public EvStatistic[] getStatisticsByTask(long task_id) throws IOException {
    return actual_gateway.getStatisticsByTask(task_id);
  }


  public void addIndividuals(Object[] individuals, long task_id,
      double[] objective_values, long cell_id, long node_id) throws IOException {
    actual_gateway.addIndividuals(individuals, task_id, objective_values,
        cell_id, node_id);
  }


  public int[] addIndividualsToEval(int task_id, long cell_id, long node_id,
      int iter, Object[] individuals) throws Exception {
    return actual_gateway.addIndividualsToEval(task_id, cell_id, node_id, iter,
        individuals);
  }


  public void addIndividualsValues(int[] ids, double[] values) throws Exception {
    actual_gateway.addIndividualsValues(ids, values);
  }


  public Object[] getIndividualsToEval(int[] ids) throws Exception {
    return actual_gateway.getIndividualsToEval(ids);
  }


  public double[] getValues(int[] ids) throws Exception {
    return actual_gateway.getValues(ids);
  }


  public void deleteIndividualsToEval() throws Exception {
    // TODO Auto-generated method stub

  }


  public void deleteIndividualsToEval(int task_id) throws Exception {

  }


  public Object[] getIndividualsToEvalByIteration(int task_id, long cell_id,
      long node_id, int iteration) throws Exception {
    // TODO Auto-generated method stub
    return null;
  }


  public void addFun(int task_id, Object fun) throws Exception {
    // TODO Auto-generated method stub

  }


  public Object getFun(int task_id) throws Exception {
    // TODO Auto-generated method stub
    return null;
  }


  public boolean presentFun(int task_id) throws Exception {
    // TODO Auto-generated method stub
    return false;
  }


  public void deleteFun(int task_id) throws Exception {
    // TODO Auto-generated method stub

  }
}
