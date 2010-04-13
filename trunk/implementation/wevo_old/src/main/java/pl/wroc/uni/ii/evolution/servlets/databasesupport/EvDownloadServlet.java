package pl.wroc.uni.ii.evolution.servlets.databasesupport;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.sql.Connection;
import java.sql.SQLException;

import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import pl.wroc.uni.ii.evolution.servlets.databasesupport.pooling.ConnectionPool;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.providers.EvDatabaseProvider;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.providers.EvDatabaseProviderFactory;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvIndividualInfo;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

/**
 * This servlet is responsible for:
 * <ul>
 * <li> downloading individuals meeting some criteria
 * <li> downloading JARs with tasks
 * <li> getting count information
 * <li> deleting individuals and tasks
 * 
 * @author Piotr Lipinski, Marcin Golebiowski
 */
public class EvDownloadServlet extends HttpServlet {

  private static final long serialVersionUID = 496915554863949071L;

  private String database_password = null;

  private String database_url = null;

  private String database_user = null;

  private String provider = null;

  private String driver = null;

  private EvDatabaseProvider database = null;

  private ConnectionPool connection_pool = null;


  private void changeTaskState(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    int id = input.readInt();
    int state = input.readInt();

    try {

      Connection conn = connection_pool.getConnection();
      database.changeTaskState(conn, id, state);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  /** *************************************************************************** */

  private void deleteIndividual(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    int individualID = input.readInt();

    try {

      Connection conn = connection_pool.getConnection();
      database.deleteIndividual(conn, individualID);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(individualID);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void deleteIndividualFromTask(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {

    long task_id = input.readLong();

    try {

      Connection conn = connection_pool.getConnection();
      database.deleteIndividualsFromTask(conn, task_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void deleteResource(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {

      String name = (String) input.readObject();

      Connection conn = connection_pool.getConnection();
      database.deleteResource(conn, name);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (IOException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void deleteSolutionSpacesForTask(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    long task_id = input.readLong();

    try {

      Connection conn = connection_pool.getConnection();
      boolean done = database.deleteSolutionSpaces(conn, task_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeBoolean(done);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void deleteTaskFromSystem(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    int id = input.readInt();

    try {

      Connection conn = connection_pool.getConnection();
      database.deleteTaskFromSystem(conn, id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }

  }


  public void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {
    ObjectInputStream input = new ObjectInputStream(request.getInputStream());
    int command = input.readInt();
    switch (command) {
      case EvDBOperations.CMD_DELETE_INDIVIDUAL_BY_ID:
        deleteIndividual(input, request, response);
        break;
      case EvDBOperations.CMD_GET_INDIVIDUAL_BY_ID:
        downloadIndividual(input, request, response);
        break;
      case EvDBOperations.CMD_GET_BEST_INDIVIDUALS:
        downloadBestIndividuals(input, request, response);
        break;
      case EvDBOperations.CMD_GET_BEST_INDIVIDUALS_MATCHING_CREATION_CELL:
        downloadBestIndividualsMatchingCreationCell(input, request, response);
        break;
      case EvDBOperations.CMD_GET_BEST_INDIVIDUALS_NOT_MATCHING_CREATION_CELL:
        downloadBestIndividualsNotMatchingCreationCell(input, request, response);
        break;
      case EvDBOperations.CMD_GET_BEST_INDIVIDUALS_MATCHING_CREATION_NODE:
        downloadBestIndividualsMatchingCreationNode(input, request, response);
        break;
      case EvDBOperations.CMD_GET_INDIVIDUAL_COUNT:
        downloadIndividualCount(input, request, response);
        break;
      case EvDBOperations.CMD_GET_INDIVIDUAL_COUNT_MATCHING_CREATION_CELL:
        downloadIndividualCountMatchingCreationCell(input, request, response);
        break;
      case EvDBOperations.CMD_GET_INDIVIDUAL_COUNT_NOT_MATCHING_CREATION_CELL:
        downloadIndividualCountNotMatchingCreationCell(input, request, response);
        break;
      case EvDBOperations.CMD_GET_INDIVIDUAL_COUNT_MATCHING_CREATION_NODE:
        downloadIndividualCountMatchingCreationNode(input, request, response);
        break;
      case EvDBOperations.CMD_DELETE_INDIVIDUAL_FROM_TASK:
        deleteIndividualFromTask(input, request, response);
        break;
      case EvDBOperations.CMD_GET_SOLUTION_SPACE:
        downloadSolutionSpace(input, request, response);
        break;

      case EvDBOperations.CMD_NEW_SPACE_VERSION_NUMBER:
        downloadNewSpaceVersion(input, request, response);
        break;

      case EvDBOperations.CMD_DELETE_SOLUTION_SPACES_FOR_TASK:
        deleteSolutionSpacesForTask(input, request, response);
        break;

      case EvDBOperations.CMD_GET_TASK_IDS:
        downloadTaskIds(input, request, response);
        break;

      case EvDBOperations.CMD_GET_TASK:
        downloadTask(input, request, response);
        break;

      case EvDBOperations.CMD_DELETE_TASK_FROM_SYSTEM:
        deleteTaskFromSystem(input, request, response);
        break;

      case EvDBOperations.CMD_GET_TASK_FOR_SYSTEM_IDS:
        downloadTaskIDsForSystem(input, request, response);
        break;

      case EvDBOperations.CMD_CHANGE_TASK_STATE_FOR_SYSTEM:
        changeTaskState(input, request, response);
        break;

      case EvDBOperations.CMD_DEL_RESOURCE:
        deleteResource(input, request, response);
        break;

      case EvDBOperations.CMD_GET_RESOURCE:
        getResource(input, request, response);
        break;
      case EvDBOperations.CMD_GET_RESOURCES_NAMES:
        getResourceNames(input, request, response);
        break;

      case EvDBOperations.CMD_GET_NODES_IDS_WITH_STATISTICS:
        getNodesIdsWithStatistics(input, request, response);
        break;

      case EvDBOperations.CMD_GET_CELL_IDS_WITH_STATISTICS:
        getCellIdsWithStatistics(input, request, response);
        break;

      case EvDBOperations.CMD_GET_STATISTICS:
        getStatistics(input, request, response);
        break;

      case EvDBOperations.CMD_GET_STATISTICS_BY_CELLS:
        getStatisticsByCells(input, request, response);
        break;

      case EvDBOperations.CMD_GET_STATISTICS_BY_NODES:
        getStatisticsByNodes(input, request, response);
        break;

      case EvDBOperations.CMD_GET_STATISTICS_BY_ID:
        getStatisticsByID(input, request, response);
        break;

      case EvDBOperations.CMD_GET_TASK_IDS_WITH_STATISTICS:
        getTaskIdsWithStatistics(input, request, response);
        break;

      case EvDBOperations.CMD_DELETE_STATISTIC_FOR_TASK:
        deleteStatisticForTask(input, request, response);
        break;

      case EvDBOperations.CMD_GET_VALUES:
        getValues(input, request, response);
        break;

      case EvDBOperations.CMD_GET_INDIVIDUALS_TO_EVAL:
        getIndividualsToEval(input, request, response);
        break;

      case EvDBOperations.CMD_DELETE_EVALUATED_INDIVIDUALS:
        deleteIndividualsToEval(input, request, response);
        break;

      case EvDBOperations.CMD_GET_INDIVIDUALS_TO_EVAL_BY_ITER:
        getIndividualsToEvalByIter(input, request, response);
        break;
      case EvDBOperations.CMD_GET_FUN:
        getFun(input, request, response);
        break;
      case EvDBOperations.CMD_CONTAIN_FUN:
        containsFun(input, request, response);
        break;
      case EvDBOperations.CMD_DELETE_FUN:
        deleteFun(input, request, response);
        break;
    }

    input.close();
  }


  private void deleteFun(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) {

    try {
      int task_id = input.readInt();

      Connection conn = connection_pool.getConnection();
      database.deleteFun(conn, task_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (Exception e) {
      e.printStackTrace();
    }
  }


  private void containsFun(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) {
    try {

      int task_id = input.readInt();

      Connection conn = connection_pool.getConnection();
      boolean res = database.presentFun(conn, task_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeBoolean(res);
      output.close();

    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }

  }


  private void getFun(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) {
    try {

      int task_id = input.readInt();
      Connection conn = connection_pool.getConnection();
      Object o = database.getFun(conn, task_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");

      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeObject(o);
      output.close();

    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }

  }


  private void getIndividualsToEvalByIter(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {

    int task_id = input.readInt();
    long cell_id = input.readLong();
    long node_id = input.readLong();
    int iteration = input.readInt();

    Object[] individuals = null;
    try {
      Connection conn = connection_pool.getConnection();
      individuals =
          database.getIndividualsToEvalByIter(conn, task_id, cell_id, node_id,
              iteration);
      connection_pool.free(conn);
    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }

    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"evolution\"");
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());

    if (individuals == null) {
      output.writeInt(0);
    } else {
      output.writeInt(individuals.length);

      for (int i = 0; i < individuals.length; i++) {
        output.writeObject(individuals[i]);
      }
    }
    output.close();
  }


  private void deleteIndividualsToEval(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    int task_id = input.readInt();
    System.err.println("asd" + task_id);
    try {
      Connection conn = connection_pool.getConnection();
      database.deleteEvaledIndividuals(conn, task_id);
      connection_pool.free(conn);
    } catch (Exception ex) {
      ex.printStackTrace();
    }

    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"evolution\"");
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());

    output.close();
  }


  private void getIndividualsToEval(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {

    int len = input.readInt();

    int[] ids = new int[len];

    for (int i = 0; i < len; i++) {
      ids[i] = input.readInt();
    }

    Object[] individuals = null;
    try {
      Connection conn = connection_pool.getConnection();
      individuals = database.getIndividualsToEval(conn, ids);
      connection_pool.free(conn);
    } catch (Exception ex) {
      ex.printStackTrace();
    }

    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"evolution\"");
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());

    output.writeInt(individuals.length);

    for (int i = 0; i < individuals.length; i++) {
      output.writeObject(individuals[i]);
    }

    output.close();

  }


  private void getValues(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {

    int len = input.readInt();

    int[] ids = new int[len];

    for (int i = 0; i < len; i++) {
      ids[i] = input.readInt();
    }

    double[] values = null;
    try {
      Connection conn = connection_pool.getConnection();
      values = database.getValues(conn, ids);
      connection_pool.free(conn);
    } catch (Exception ex) {
      ex.printStackTrace();
    }

    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"evolution\"");
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());

    output.writeInt(values.length);

    for (int i = 0; i < values.length; i++) {
      output.writeDouble(values[i]);
    }

    output.close();
  }


  private void deleteStatisticForTask(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {

    try {
      long task_id = input.readLong();

      Connection conn = connection_pool.getConnection();
      database.deleteStatisticForTask(conn, task_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  private void downloadBestIndividuals(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    long instanceID = input.readLong();
    int k = input.readInt();
    int n = input.readInt();
    boolean withIndividualsItselves = input.readBoolean();

    try {

      Connection conn = connection_pool.getConnection();
      EvIndividualInfo[] infos =
          database.getBestIndividualInfos(conn, instanceID, k, n,
              withIndividualsItselves);
      connection_pool.free(conn);

      int m = 0;
      for (int i = 0; i < infos.length; i++) {

        if (infos[i] != null) {
          m++;
        }
      }

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(m);
      for (int i = 0; i < infos.length; i++)
        if (infos[i] != null)
          saveIndividualInfo(output, infos[i], withIndividualsItselves);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadBestIndividualsMatchingCreationCell(
      ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    long instanceID = input.readLong();
    long creationCell = input.readLong();
    int k = input.readInt();
    int n = input.readInt();
    boolean withIndividualsItselves = input.readBoolean();

    try {

      Connection conn = connection_pool.getConnection();
      EvIndividualInfo[] infos =
          database.getBestIndividualInfosMatchingCell(conn, instanceID,
              creationCell, k, n, withIndividualsItselves);
      connection_pool.free(conn);

      int m = 0;
      for (int i = 0; i < infos.length; i++) {
        if (infos[i] != null) {
          m++;
        }
      }

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(m);
      for (int i = 0; i < infos.length; i++)
        if (infos[i] != null)
          saveIndividualInfo(output, infos[i], withIndividualsItselves);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadBestIndividualsMatchingCreationNode(
      ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    long instanceID = input.readLong();
    long creationNode = input.readLong();
    int k = input.readInt();
    int n = input.readInt();
    boolean withIndividualsItselves = input.readBoolean();

    try {

      Connection conn = connection_pool.getConnection();
      EvIndividualInfo[] infos =
          database.getBestIndividualInfosMatchingNode(conn, instanceID,
              creationNode, k, n, withIndividualsItselves);
      connection_pool.free(conn);

      int m = 0;
      for (int i = 0; i < infos.length; i++) {
        if (infos[i] != null) {
          m++;
        }
      }

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(m);
      for (int i = 0; i < infos.length; i++)
        if (infos[i] != null)
          saveIndividualInfo(output, infos[i], withIndividualsItselves);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadBestIndividualsNotMatchingCreationCell(
      ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    long instanceID = input.readLong();
    long creationCell = input.readLong();
    int k = input.readInt();
    int n = input.readInt();
    boolean withIndividualsItselves = input.readBoolean();

    try {

      Connection conn = connection_pool.getConnection();
      EvIndividualInfo[] infos =
          database.getBestIndividualInfosNotMatchingCell(conn, instanceID,
              creationCell, k, n, withIndividualsItselves);
      connection_pool.free(conn);

      int m = 0;
      for (int i = 0; i < infos.length; i++) {
        if (infos[i] != null) {
          m++;
        }
      }

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(m);
      for (int i = 0; i < infos.length; i++)
        if (infos[i] != null)
          saveIndividualInfo(output, infos[i], withIndividualsItselves);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadIndividual(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {

    int individualID = input.readInt();
    boolean withIndividualItself = input.readBoolean();

    try {

      Connection conn = connection_pool.getConnection();
      EvIndividualInfo info =
          database.getIndividualInfo(conn, individualID, withIndividualItself);
      connection_pool.free(conn);

      if (info == null) {
        System.out.println(info + "is null");
      }

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      saveIndividualInfo(output, info, withIndividualItself);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  /** *************************************************************************** */

  private void downloadIndividualCount(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    long instanceID = input.readLong();

    try {

      Connection conn = connection_pool.getConnection();
      int count = database.getIndividualCount(conn, instanceID);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(count);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadIndividualCountMatchingCreationCell(
      ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    long instanceID = input.readLong();
    long creationCell = input.readLong();

    try {

      Connection conn = connection_pool.getConnection();
      int count =
          database.getIndividualCountMatchingCell(conn, instanceID,
              creationCell);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(count);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadIndividualCountMatchingCreationNode(
      ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    long instanceID = input.readLong();
    long creationNode = input.readLong();

    try {

      Connection conn = connection_pool.getConnection();
      int count =
          database.getIndividualCountMatchingNode(conn, instanceID,
              creationNode);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(count);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadIndividualCountNotMatchingCreationCell(
      ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    long instanceID = input.readLong();
    long creationCell = input.readLong();

    try {

      Connection conn = connection_pool.getConnection();
      int count =
          database.getIndividualCountNotMatchingCell(conn, instanceID,
              creationCell);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(count);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadNewSpaceVersion(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    long task_id = input.readLong();
    long cell_id = input.readLong();

    try {

      Connection conn = connection_pool.getConnection();
      int count = database.getVersionOfNewSolutonSpace(conn, task_id, cell_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(count);
      output.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadSolutionSpace(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    try {
      long task_id = input.readLong();
      long cell_id = input.readLong();

      Connection conn = connection_pool.getConnection();
      Object space = database.getSolutionSpace(conn, task_id, cell_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeBoolean(space != null);
      output.writeObject(space);
      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadTask(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {

    try {
      int id = input.readInt();

      Connection conn = connection_pool.getConnection();
      EvTaskInfo info = database.getTaskForSystem(conn, id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (info != null) {
        output.writeBoolean(true);
        output.writeInt(info.getId());
        output.writeInt(info.getStatus());
        output.writeObject(info.getDescription());
        output.writeObject(info.getSubmissionTime());
        output.writeObject(info.getJar());
      } else {
        output.writeBoolean(false);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadTaskIds(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {

    try {

      Connection conn = connection_pool.getConnection();
      Long[] ids = database.getTaskIDs(conn);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeLong(ids[i]);
        }
      } else {
        output.writeInt(0);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void downloadTaskIDsForSystem(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    try {

      Connection conn = connection_pool.getConnection();
      Integer[] ids = database.getTaskIDsForSystem(conn);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeInt(ids[i]);
        }
      } else {
        output.writeInt(0);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void getCellIdsWithStatistics(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {

      long task_id = input.readLong();

      Connection conn = connection_pool.getConnection();
      Long[] ids = database.getCellIdsWithStatistics(conn, task_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeLong(ids[i]);
        }
      } else {
        output.writeInt(0);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  private void getNodesIdsWithStatistics(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {

    try {

      long task_id = input.readLong();
      long cell_id = input.readLong();

      Connection conn = connection_pool.getConnection();
      Long[] ids = database.getNodesIdsWithStatistics(conn, task_id, cell_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeLong(ids[i]);
        }
      } else {
        output.writeInt(0);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  private void getResource(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) {

    try {
      String name = (String) input.readObject();

      Connection conn = connection_pool.getConnection();
      Object res = database.getResource(conn, name);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeBoolean(res != null);
      output.writeObject(res);
      output.flush();
      output.close();
    } catch (Exception ex) {
      ex.printStackTrace();
    }

  }


  private void getResourceNames(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {

      Connection conn = connection_pool.getConnection();
      String[] names = database.getResourceNames(conn);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeObject(names);
      output.close();
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }


  private void getStatistics(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {

      long task_id = input.readLong();
      long cell_id = input.readLong();
      long node_id = input.readLong();

      Connection conn = connection_pool.getConnection();
      Object[] ids = database.getStatistics(conn, task_id, cell_id, node_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeObject(ids[i]);
        }
      } else {
        output.writeInt(0);
      }
      output.close();
    } catch (SQLException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  private void getStatisticsByCells(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {

      long task_id = input.readLong();
      int cells_number = input.readInt();
      int[] cells = new int[cells_number];

      for (int i = 0; i < cells_number; i++) {
        cells[i] = input.readInt();
      }

      Connection conn = connection_pool.getConnection();
      Object[] ids = database.getStatisticsByCells(conn, task_id, cells);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeObject(ids[i]);
        }
      } else {
        output.writeInt(0);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }

  }


  private void getStatisticsByNodes(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {

      long task_id = input.readLong();
      int nodes_number = input.readInt();
      int[] nodes = new int[nodes_number];

      for (int i = 0; i < nodes_number; i++) {
        nodes[i] = input.readInt();
      }

      Connection conn = connection_pool.getConnection();
      Object[] ids = database.getStatisticsByNodes(conn, task_id, nodes);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeObject(ids[i]);
        }
      } else {
        output.writeInt(0);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }

  }


  private void getStatisticsByID(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {

      long task_id = input.readLong();

      Connection conn = connection_pool.getConnection();
      Object[] ids = database.getStatisticsByID(conn, task_id);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeObject(ids[i]);
        }
      } else {
        output.writeInt(0);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }

  }


  private void getTaskIdsWithStatistics(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {
      Connection conn = connection_pool.getConnection();
      Long[] ids = database.getTaskIdsWithStatistics(conn);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      if (ids != null) {
        output.writeInt(ids.length);

        for (int i = 0; i < ids.length; i++) {
          output.writeLong(ids[i]);
        }
      } else {
        output.writeInt(0);
      }

      output.close();

    } catch (SQLException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }


  /** *************************************************************************** */

  public void init(ServletConfig config) throws ServletException {
    super.init(config);

    database_url = config.getInitParameter("DATABASE_URL");
    database_password = config.getInitParameter("DATABASE_PASSWORD");
    database_user = config.getInitParameter("DATABASE_USER");
    provider = config.getInitParameter("DATABASE_PROVIDER");
    driver = config.getInitParameter("DATABASE_DRIVER");

    try {
      database = EvDatabaseProviderFactory.getProvider(provider);
      database.init();

      connection_pool =
          new ConnectionPool(driver, database_url, database_user,
              database_password, 10, 100, true);
    } catch (SQLException sqle) {
      System.err.println("Error making pool: " + sqle);
      getServletContext().log("Error making pool: " + sqle);
    } catch (InstantiationException e) {
      e.printStackTrace();
    } catch (IllegalAccessException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
  }


  /** *************************************************************************** */

  private void saveIndividualInfo(ObjectOutputStream output,
      EvIndividualInfo info, boolean withIndividualItself) throws IOException {
    output.writeInt(info.getID());
    output.writeLong(info.getTaskID());
    output.writeDouble(info.getObjectiveValue());
    output.writeLong(info.getCellID());
    output.writeLong(info.getNodeID());
    output.writeObject(info.getCreationTime());
    if (withIndividualItself)
      output.writeObject(info.getIndividual());
  }


  public void destroy() {
    connection_pool.closeAllConnections();
  }
}