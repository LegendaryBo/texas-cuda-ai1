package pl.wroc.uni.ii.evolution.sampleimplementation.tools;

import java.io.IOException;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.totaltime.EvTotalComputationTimeStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.bestindividual.EvBestIndividualStatistic;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

/**
 * In future it could be usefull for others. It will be tested in not too
 * distant future.
 * 
 * @author Tomasz Kozakiewicz
 */
public class EvSummariesOperations {

  private static EvDBServletCommunicationImpl db_comm;


  public static void main(String[] args) {
    Long[] tasks = null;

    initCommunication();
    tasks = getTasksIDs(db_comm);

    if (tasks != null) {
      for (int h = 0; h < tasks.length; h++) {
        summaryOneTask(tasks[h]);
      }
    }
  }


  private static void initCommunication() {
    String wevo_url = "http://192.168.0.3:8080";

    db_comm = new EvDBServletCommunicationImpl(wevo_url);
  }


  private static void summaryOneTask(long task_id) {
    EvTaskInfo task_info = null;
    Long[] cells = null;
    ComputationResult bestMaxMinAvgStddev;
    ComputationResult lastMaxMinAvgStddev;
    ComputationResult bestIndividual;
    ComputationResult lastIndividual;
    long computation_time;
    int total_iterations_number;

    System.out.println("Task " + task_id + " proceeding...");

    try {
      task_info = db_comm.getTaskForSystem((int) task_id);
    } catch (IOException e) {
      e.printStackTrace();
    }

    if (task_info != null) {
      System.out.println("Task description: " + task_info.getDescription());
    } else {
      System.out.println("Task probably deleted.");
    }

    cells = getCellsID(task_id);

    bestMaxMinAvgStddev = getBestResultOfMaxAvgMinStddev(task_id, cells);
    lastMaxMinAvgStddev = getLastResultOfMaxAvgMinStddev(task_id, cells);

    bestIndividual = getBestResultIndividual(task_id, cells);
    lastIndividual = getLastResultIndividual(task_id, cells);

    computation_time = getTotalComputationTime(task_id, cells);
    total_iterations_number = getTotalIterationsNumber(task_id, cells);

    System.out.println("Best's result: " + bestMaxMinAvgStddev);
    System.out.println("Best individaul: " + bestIndividual);

    System.out.println("Last completed algorithm's result: "
        + lastMaxMinAvgStddev);
    System.out.println("Last completed algorithm's individual: "
        + lastIndividual);

    if (computation_time > 0) {
      System.out.println("Total computation time: " + computation_time / 1000
          + " seconds.");
    }

    System.out.println("Total iterations number: " + total_iterations_number);
    System.out.println();
    // EvIndividualInfo[] individuals_infos = null;
    // try {
    // individuals_infos
    // = db_comm.getBestIndividualInfosMatchingNode(task_id, nodes[0], 1, 1,
    // true);
    // } catch (IOException e) {
    // e.printStackTrace();
    // }

    // db_comm.getBestIndividualInfosMatchingCell(task_id, creationCell, k, n,
    // with_individuals_itselves);
    // EvStatistic[][] stats_tab = stats.toArray(new
    // EvStatistic[stats.size()][]);
  }


  private static ComputationResult getBestResultOfMaxAvgMinStddev(long task_id,
      Long[] cells) {
    return getBestResult(task_id, cells, false);
  }


  private static ComputationResult getBestResultIndividual(long task_id,
      Long[] cells) {
    return getBestResult(task_id, cells, true);
  }


  private static ComputationResult getBestResult(long task_id, Long[] cells,
      boolean getWholeIndividual) {

    ComputationResult best = new ComputationResult();
    EvStatistic[] node_stats = null;

    Long[] nodes;

    for (int i = 0; i < cells.length; i++) {
      nodes = null;
      try {
        nodes = db_comm.getNodesIdsWithStatistics(task_id, cells[i]);
      } catch (IOException e) {
        e.printStackTrace();
      }

      if (nodes != null) {
        for (int j = 0; j < nodes.length; j++) {
          try {
            if (getWholeIndividual) {
              node_stats =
                  EvStatisticFilter.byClass(EvBestIndividualStatistic.class,
                      db_comm.getStatistics(task_id, cells[i], nodes[j]));
            } else {
              node_stats =
                  EvStatisticFilter.byClass(
                      EvObjectiveFunctionValueMaxAvgMinStatistic.class, db_comm
                          .getStatistics(task_id, cells[i], nodes[j]));
            }
            if (node_stats != null) {
              for (int k = 0; k < node_stats.length; k++) {
                EvStatistic curr_stat = node_stats[k];

                if (getWholeIndividual) {
                  EvBestIndividualStatistic casted_curr_stat =
                      (EvBestIndividualStatistic) curr_stat;
                  if (best.isEmpty()
                      || casted_curr_stat.getObjectiveFunctionValue() > best
                          .getObjectiveFunctionValue()) {
                    best =
                        makeResult(task_id, cells[i], nodes[j],
                            casted_curr_stat);
                  }
                } else {
                  EvObjectiveFunctionValueMaxAvgMinStatistic casted_curr_stat =
                      (EvObjectiveFunctionValueMaxAvgMinStatistic) curr_stat;
                  if (best.isEmpty()
                      || casted_curr_stat.getMax() > best.getMax()) {
                    best =
                        makeResult(task_id, cells[i], nodes[j],
                            casted_curr_stat);
                  }
                }
              }
            }
          } catch (IOException e) {
            e.printStackTrace();
          }
        }
      }
    }
    return best;
  }


  private static ComputationResult getLastResultOfMaxAvgMinStddev(long task_id,
      Long[] cells) {
    return getLastResult(task_id, cells, false);
  }


  private static ComputationResult getLastResultIndividual(long task_id,
      Long[] cells) {
    return getLastResult(task_id, cells, true);
  }


  /**
   * Gets result of last completed algorithm. (In details: gets result of
   * iteration with greatest number and if there is many of them, then result
   * with the most recent time.)
   */
  private static ComputationResult getLastResult(long task_id, Long[] cells,
      boolean getWholeIndividual) {
    ComputationResult last = new ComputationResult();
    long shared_last_stat_time = 0;
    int shared_last_iteration_nr = 0;
    EvStatistic[] node_stats = null;
    Long[] nodes;

    for (int i = 0; i < cells.length; i++) {
      nodes = null;
      try {
        nodes = db_comm.getNodesIdsWithStatistics(task_id, cells[i]);
      } catch (IOException e) {
        e.printStackTrace();
      }

      if (nodes != null) {
        for (int j = 0; j < nodes.length; j++) {
          try {
            if (getWholeIndividual) {
              node_stats =
                  EvStatisticFilter.byClass(EvBestIndividualStatistic.class,
                      db_comm.getStatistics(task_id, cells[i], nodes[j]));
            } else {
              node_stats =
                  EvStatisticFilter.byClass(
                      EvObjectiveFunctionValueMaxAvgMinStatistic.class, db_comm
                          .getStatistics(task_id, cells[i], nodes[j]));
            }
            if (node_stats != null) {
              for (int k = 0; k < node_stats.length; k++) {
                EvStatistic curr_stat = node_stats[k];

                if (shared_last_iteration_nr < curr_stat.getIteration()) {
                  last =
                      makeResult(task_id, cells[i], nodes[j], curr_stat,
                          getWholeIndividual);
                  shared_last_iteration_nr = curr_stat.getIteration();
                }

                if (shared_last_stat_time < curr_stat.getTime()
                    && shared_last_iteration_nr == curr_stat.getIteration()) {
                  last =
                      makeResult(task_id, cells[i], nodes[j], curr_stat,
                          getWholeIndividual);
                  shared_last_stat_time = curr_stat.getTime();
                }
              }
            }
          } catch (IOException e) {
            e.printStackTrace();
          }
        }
      }
    }
    return last;
  }


  private static ComputationResult makeResult(long task_id, long cell_id,
      long node_id, EvStatistic stat, boolean getWholeIndividual) {
    if (getWholeIndividual) {
      return makeResult(task_id, cell_id, node_id,
          (EvBestIndividualStatistic) stat);
    } else {
      return makeResult(task_id, cell_id, node_id,
          (EvObjectiveFunctionValueMaxAvgMinStatistic) stat);
    }
  }


  private static ComputationResult makeResult(long task_id, long cell_id,
      long node_id, EvObjectiveFunctionValueMaxAvgMinStatistic stat) {
    return new ComputationResult(task_id, cell_id, node_id, stat.getMax(), stat
        .getMin(), stat.getAvg(), stat.getStdev());
  }


  private static ComputationResult makeResult(long task_id, long cell_id,
      long node_id, EvBestIndividualStatistic stat) {
    EvBinaryVectorIndividual individual =
        new EvBinaryVectorIndividual(stat.getBits());
    return new ComputationResult(task_id, cell_id, node_id, individual, stat
        .getObjectiveFunctionValue());
  }


  private static long getTotalComputationTime(long task_id, Long[] cells) {
    EvStatistic[] node_stats;
    Long[] nodes;
    long total_computation_time = 0;
    long first_iteration_time;
    long last_iteration_time;
    EvTotalComputationTimeStatistic curr_stat;

    for (int i = 0; i < cells.length; i++) {
      nodes = null;
      try {
        nodes = db_comm.getNodesIdsWithStatistics(task_id, cells[i]);
      } catch (IOException e) {
        e.printStackTrace();
      }

      if (nodes != null) {
        for (int j = 0; j < nodes.length; j++) {
          try {
            node_stats =
                EvStatisticFilter.byClass(
                    EvTotalComputationTimeStatistic.class, db_comm
                        .getStatistics(task_id, cells[i], nodes[j]));

            if (node_stats != null && node_stats.length > 0) {
              curr_stat = (EvTotalComputationTimeStatistic) node_stats[0];
              first_iteration_time = curr_stat.getTimeOnNode();
              last_iteration_time = first_iteration_time;

              for (int k = 1; k < node_stats.length; k++) {
                curr_stat = (EvTotalComputationTimeStatistic) node_stats[k];

                last_iteration_time = curr_stat.getTime();
                if (last_iteration_time < first_iteration_time) {
                  throw new IllegalStateException(
                      "Statistics are not sorted by time.");
                }
              }
              total_computation_time +=
                  last_iteration_time - first_iteration_time;
            }
          } catch (IOException e) {
            e.printStackTrace();
          }
        }
      }
    }
    return total_computation_time;
  }


  /**
   * This method uses EvObjectiveFunctionValueMaxAvgMinStatistic for counting.
   */
  private static int getTotalIterationsNumber(long task_id, Long[] cells) {
    int total_iteration_number = 0;
    EvStatistic[] node_stats = null;
    Long[] nodes;

    for (int i = 0; i < cells.length; i++) {
      nodes = null;
      try {
        nodes = db_comm.getNodesIdsWithStatistics(task_id, cells[i]);
      } catch (IOException e) {
        e.printStackTrace();
      }

      if (nodes != null) {
        for (int j = 0; j < nodes.length; j++) {
          try {
            node_stats =
                EvStatisticFilter.byClass(
                    EvObjectiveFunctionValueMaxAvgMinStatistic.class, db_comm
                        .getStatistics(task_id, cells[i], nodes[j]));
            total_iteration_number += node_stats.length;
          } catch (IOException e) {
            e.printStackTrace();
          }
        }
      }
    }
    return total_iteration_number;
  }


  private static Long[] getTasksIDs(EvDBServletCommunicationImpl db_comm) {
    Long[] tasks = null;
    try {
      tasks = db_comm.getTaskIdsWithStatistics();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return tasks;
  }


  private static Long[] getCellsID(long task_id) {
    Long[] cells = null;
    try {
      cells = db_comm.getCellIdsWithStatistics(task_id);
    } catch (IOException e1) {
      e1.printStackTrace();
    }
    return cells;
  }
}

class ComputationResult {
  private double value;

  private double max, min, avg, stddev;

  private EvIndividual individual;

  private long task;

  private long cell;

  private long node;

  private double objective_function_value; // TODO temporary

  private enum ResultType {
    SINGLEVALUE, DETAILEDVALUES, WHOLEINDIVIDUAL, NORESULTS
  };

  private ResultType result_type;


  public ComputationResult() {
    result_type = ResultType.NORESULTS;
  }


  public ComputationResult(long task, long cell, long node, double value) {
    setResult(task, cell, node, value);
  }


  public ComputationResult(long task, long cell, long node, double max,
      double min, double avg, double stddev) {
    setResult(task, cell, node, max, min, avg, stddev);
  }


  public ComputationResult(long task, long cell, long node,
      EvIndividual individual, double objective_function_value) {
    setResult(task, cell, node, individual, objective_function_value);
  }


  public long getTask() {
    return task;
  }


  public long getCell() {
    return cell;
  }


  public long getNode() {
    return node;
  }


  public double getAvg() {
    if (result_type == ResultType.DETAILEDVALUES) {
      return avg;
    } else {
      throw new IllegalStateException("Average haven't been set.");
    }
  }


  public double getMax() {
    if (result_type == ResultType.DETAILEDVALUES) {
      return max;
    } else {
      throw new IllegalStateException("Maximum haven't been set.");
    }
  }


  public double getMin() {
    if (result_type == ResultType.DETAILEDVALUES) {
      return min;
    } else {
      throw new IllegalStateException("Minimum haven't been set.");
    }
  }


  public double getStddev() {
    if (result_type == ResultType.DETAILEDVALUES) {
      return stddev;
    } else {
      throw new IllegalStateException("Standard deviation haven't been set.");
    }
  }


  public double getValue() {
    if (result_type == ResultType.SINGLEVALUE) {
      return value;
    } else {
      throw new IllegalStateException("Value haven't been set.");
    }
  }


  public EvIndividual getIndividual() {
    if (result_type == ResultType.WHOLEINDIVIDUAL) {
      return individual;
    } else {
      throw new IllegalStateException("Individual haven't been set.");
    }
  }


  public double getObjectiveFunctionValue() {
    if (result_type == ResultType.WHOLEINDIVIDUAL) {
      return objective_function_value;
    } else {
      throw new IllegalStateException(
          "Objective function value haven't been set.");
    }
  }


  // public void setResult(long cell, long node, double value) {
  // setResult(-1, cell, node, value);
  // }

  public void setResult(long task, long cell, long node, double value) {
    this.task = task;
    this.cell = cell;
    this.node = node;
    this.value = value;
    this.result_type = ResultType.SINGLEVALUE;
  }


  // public void setResult(long cell, long node, double max, double min, double
  // avg, double stddev) {
  // setResult(-1, cell, node, max, min, avg, stddev);
  // }

  public void setResult(long task, long cell, long node, double max,
      double min, double avg, double stddev) {
    this.task = task;
    this.cell = cell;
    this.node = node;
    this.max = max;
    this.min = min;
    this.avg = avg;
    this.stddev = stddev;
    this.result_type = ResultType.DETAILEDVALUES;
  }


  public void setResult(long task, long cell, long node,
      EvIndividual individual, double objective_function_value) {
    this.task = task;
    this.cell = cell;
    this.node = node;
    this.individual = individual; // TODO 2 clone or not 2 clone
    this.objective_function_value = objective_function_value;
    this.result_type = ResultType.WHOLEINDIVIDUAL;
  }


  public boolean isEmpty() {
    if (result_type == ResultType.NORESULTS) {
      return true;
    } else {
      return false;
    }
  }


  public String toString() {
    if (result_type == ResultType.SINGLEVALUE) {
      return "value = " + value + " (task: " + task + ", cell: " + cell
          + ", node: " + node + ")";
    }
    if (result_type == ResultType.DETAILEDVALUES) {
      return "max = " + max + ", min = " + min + ", avg = " + avg
          + ", stddev = " + stddev + " (task: " + task + ", cell: " + cell
          + ", node: " + node + ")";
    }
    if (result_type == ResultType.WHOLEINDIVIDUAL) {
      return individual.toString();
    } else {
      return null;
    }
  }

}
