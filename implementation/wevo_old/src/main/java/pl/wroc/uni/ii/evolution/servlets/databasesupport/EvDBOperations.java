package pl.wroc.uni.ii.evolution.servlets.databasesupport;

public class EvDBOperations {
  public final static int CMD_DELETE_INDIVIDUAL_BY_ID = 1;

  public final static int CMD_GET_INDIVIDUAL_BY_ID = 2;

  public final static int CMD_GET_BEST_INDIVIDUALS = 3;

  public final static int CMD_GET_BEST_INDIVIDUALS_MATCHING_CREATION_CELL = 4;

  public final static int CMD_GET_BEST_INDIVIDUALS_NOT_MATCHING_CREATION_CELL =
      5;

  public final static int CMD_GET_BEST_INDIVIDUALS_MATCHING_CREATION_NODE = 6;

  public final static int CMD_GET_INDIVIDUAL_COUNT = 7;

  public final static int CMD_GET_INDIVIDUAL_COUNT_MATCHING_CREATION_CELL = 8;

  public final static int CMD_GET_INDIVIDUAL_COUNT_NOT_MATCHING_CREATION_CELL =
      9;

  public final static int CMD_GET_INDIVIDUAL_COUNT_MATCHING_CREATION_NODE = 10;

  public final static int CMD_GET_SOLUTION_SPACE = 11;

  public final static int CMD_DELETE_INDIVIDUAL_FROM_TASK = 12;

  public final static int CMD_DELETE_SOLUTION_SPACES_FOR_TASK = 13;

  public final static int CMD_ADD_SOLUTION_SPACES_FOR_TASK_AND_CELL = 14;

  public final static int CMD_ADD_INDIVIDUAL = 15;

  public final static int CMD_GET_TASK_IDS = 16;

  public final static int CMD_NEW_SPACE_VERSION_NUMBER = 17;

  public final static int CMD_ADD_TASK = 18;

  public final static int CMD_GET_TASK = 19;

  public final static int CMD_GET_TASK_FOR_SYSTEM_IDS = 20;

  public final static int CMD_DELETE_TASK_FROM_SYSTEM = 21;

  public final static int CMD_CHANGE_TASK_STATE_FOR_SYSTEM = 22;

  public final static int CMD_SET_RESOURCE = 23;

  public final static int CMD_GET_RESOURCE = 24;

  public final static int CMD_DEL_RESOURCE = 25;

  public final static int CMD_GET_RESOURCES_NAMES = 26;

  public final static int CMD_DELETE_STATISTIC_FOR_TASK = 27;

  public final static int CMD_GET_CELL_IDS_WITH_STATISTICS = 28;

  public final static int CMD_GET_NODES_IDS_WITH_STATISTICS = 29;

  public final static int CMD_GET_STATISTICS = 30;

  public final static int CMD_GET_TASK_IDS_WITH_STATISTICS = 31;

  public final static int CMD_SAVE_STATISTICS = 32;

  public final static int CMD_GET_STATISTICS_BY_CELLS = 33;

  public final static int CMD_GET_STATISTICS_BY_NODES = 34;

  public final static int CMD_GET_STATISTICS_BY_ID = 35;

  public final static int CMD_ADD_INDIVIDUALS = 36;

  public final static int CMD_ADD_INDIVIDUALS_TO_EVAL = 37;

  public final static int CMD_ADD_INDIVIDUALS_VALUES = 38;

  public final static int CMD_GET_INDIVIDUALS_TO_EVAL = 39;

  public final static int CMD_GET_VALUES = 40;

  public final static int CMD_DELETE_EVALUATED_INDIVIDUALS = 41;

  public final static int CMD_GET_INDIVIDUALS_TO_EVAL_BY_ITER = 42;

  public final static int CMD_ADD_FUN = 43;

  public final static int CMD_GET_FUN = 44;

  public final static int CMD_CONTAIN_FUN = 45;

  public final static int CMD_DELETE_FUN = 46;

}