package pl.wroc.uni.ii.evolution.servlets.databasesupport.communication;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URL;
import java.net.URLConnection;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.distribution.tools.EvDistributionTools;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.EvDBOperations;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvIndividualInfo;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

/**
 * @author Piotr Lipinski, Marcin Golebiowski
 */
public class EvDBServletCommunicationImpl implements EvDBServletCommunication {

  private String DOWNLOAD_SERVLET_ADDRESS;

  private String UPLOAD_SERVLET_ADDRESS;


  /**
   * @param wevo_server_url - url to a computer with wEvo installed on it
   */
  public EvDBServletCommunicationImpl(String wevo_server_url) {
    this.DOWNLOAD_SERVLET_ADDRESS =
        EvDistributionTools.download_servlet_url(wevo_server_url);
    this.UPLOAD_SERVLET_ADDRESS =
        EvDistributionTools.upload_servlet_url(wevo_server_url);
    ;
  }


  /**
   * @deprecated - use EvDBServletCommunicationImpl(String wevo_server_url)
   *             instead
   */
  public EvDBServletCommunicationImpl(String download, String upload) {
    this.DOWNLOAD_SERVLET_ADDRESS = download;
    this.UPLOAD_SERVLET_ADDRESS = upload;
  }


  /** *************************************************************************** */

  private byte[] toByteArray(Object ob) {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try {
      ObjectOutputStream os = new ObjectOutputStream(baos);
      os.writeObject(ob);
      return baos.toByteArray();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return null;
  }


  private Object fromByteArray(Object ob) {
    ByteArrayInputStream baos = new ByteArrayInputStream((byte[]) ob);
    try {
      ObjectInputStream os = new ObjectInputStream(baos);
      return os.readObject();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
    return null;
  }


  /* ============================================================ */

  public int addIndividual(Object individual, long task_id,
      double objectiveValue, long cell_id, long node_id) throws IOException {
    int ID = -1;

    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_ADD_INDIVIDUAL);
    output.writeLong(task_id);
    output.writeDouble(objectiveValue);
    output.writeObject(toByteArray(individual));
    output.writeLong(cell_id);
    output.writeLong(node_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    ID = input.readInt();
    input.close();

    return ID;
  }


  /** *************************************************************************** */

  public boolean deleteIndividual(int id) throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_DELETE_INDIVIDUAL_BY_ID);
    output.writeInt(id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());

    boolean done = input.readBoolean();
    input.close();

    return done;
  }


  private EvIndividualInfo loadIndividualInfo(ObjectInputStream input,
      boolean with_individual_itself) throws IOException,
      ClassNotFoundException {
    EvIndividualInfo info = new EvIndividualInfo();

    info.setID(input.readInt());
    info.setTaskID(input.readLong());
    info.setObjectiveValue(input.readDouble());
    info.setCellID(input.readLong());
    info.setNodeID(input.readLong());
    info.setCreationTime((Timestamp) input.readObject());
    if (with_individual_itself)
      info.setIndividual(fromByteArray(input.readObject()));

    return info;
  }


  public EvIndividualInfo getIndividualInfo(int ID,
      boolean with_individual_itself) throws IOException {
    EvIndividualInfo info = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output.writeInt(EvDBOperations.CMD_GET_INDIVIDUAL_BY_ID);
      output.writeInt(ID);
      output.writeBoolean(with_individual_itself);
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      info = loadIndividualInfo(input, with_individual_itself);
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return info;
  }


  public EvIndividualInfo[] getBestIndividualInfos(long task_id, int k, int n,
      boolean with_individuals_itselves) throws IOException {
    EvIndividualInfo[] infos = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output.writeInt(EvDBOperations.CMD_GET_BEST_INDIVIDUALS);
      output.writeLong(task_id);
      output.writeInt(k);
      output.writeInt(n);
      output.writeBoolean(with_individuals_itselves);
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      int m = input.readInt();
      infos = new EvIndividualInfo[m];
      for (int i = 0; i < m; i++) {
        infos[i] = loadIndividualInfo(input, with_individuals_itselves);
      }
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return infos;
  }


  public EvIndividualInfo[] getBestIndividualInfosMatchingCell(long task_id,
      long creationCell, int k, int n, boolean with_individuals_itselves)
      throws IOException {
    EvIndividualInfo[] infos = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output
          .writeInt(EvDBOperations.CMD_GET_BEST_INDIVIDUALS_MATCHING_CREATION_CELL);
      output.writeLong(task_id);
      output.writeLong(creationCell);
      output.writeInt(k);
      output.writeInt(n);
      output.writeBoolean(with_individuals_itselves);
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      int m = input.readInt();
      infos = new EvIndividualInfo[m];

      for (int i = 0; i < m; i++) {
        infos[i] = loadIndividualInfo(input, with_individuals_itselves);
      }
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return infos;
  }


  public EvIndividualInfo[] getBestIndividualInfosNotMatchingCell(long task_id,
      long creationCell, int k, int n, boolean with_individuals_itselves)
      throws IOException {
    EvIndividualInfo[] infos = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output
          .writeInt(EvDBOperations.CMD_GET_BEST_INDIVIDUALS_NOT_MATCHING_CREATION_CELL);
      output.writeLong(task_id);
      output.writeLong(creationCell);
      output.writeInt(k);
      output.writeInt(n);
      output.writeBoolean(with_individuals_itselves);
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      int m = input.readInt();
      infos = new EvIndividualInfo[m];
      for (int i = 0; i < m; i++)
        infos[i] = loadIndividualInfo(input, with_individuals_itselves);
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return infos;
  }


  public EvIndividualInfo[] getBestIndividualInfosMatchingNode(long task_id,
      long node_id, int k, int n, boolean with_individuals_itselves)
      throws IOException {
    EvIndividualInfo[] infos = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output
          .writeInt(EvDBOperations.CMD_GET_BEST_INDIVIDUALS_MATCHING_CREATION_NODE);
      output.writeLong(task_id);
      output.writeLong(node_id);
      output.writeInt(k);
      output.writeInt(n);
      output.writeBoolean(with_individuals_itselves);
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      int m = input.readInt();
      infos = new EvIndividualInfo[m];
      for (int i = 0; i < m; i++)
        infos[i] = loadIndividualInfo(input, with_individuals_itselves);
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return infos;
  }


  /** *************************************************************************** */

  public int getIndividualCount(long task_id) throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_INDIVIDUAL_COUNT);
    output.writeLong(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int count = input.readInt();
    input.close();

    return count;
  }


  public int getIndividualCountMatchingCell(long task_id, long cell_id)
      throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output
        .writeInt(EvDBOperations.CMD_GET_INDIVIDUAL_COUNT_MATCHING_CREATION_CELL);
    output.writeLong(task_id);
    output.writeLong(cell_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int count = input.readInt();
    input.close();

    return count;
  }


  public int getIndividualCountNotMatchingCell(long task_id, long cell_id)
      throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output
        .writeInt(EvDBOperations.CMD_GET_INDIVIDUAL_COUNT_NOT_MATCHING_CREATION_CELL);
    output.writeLong(task_id);
    output.writeLong(cell_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int count = input.readInt();
    input.close();

    return count;
  }


  public int getIndividualCountMatchingNode(long task_id, long node_id)
      throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output
        .writeInt(EvDBOperations.CMD_GET_INDIVIDUAL_COUNT_MATCHING_CREATION_NODE);
    output.writeLong(task_id);
    output.writeLong(node_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int count = input.readInt();
    input.close();

    return count;
  }


  public void deleteIndividualsFromTask(long task_id) throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_DELETE_INDIVIDUAL_FROM_TASK);
    output.writeLong(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();

  }


  public Object getSolutionSpace(long task_id, long cell_id) throws IOException {
    Object space = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output.writeInt(EvDBOperations.CMD_GET_SOLUTION_SPACE);
      output.writeLong(task_id);
      output.writeLong(cell_id);
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());

      if (input.readBoolean()) {
        space = fromByteArray(input.readObject());
      }
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return space;
  }


  public int addSolutionSpace(long task_id, long cell_id, Object space)
      throws IOException {

    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_ADD_SOLUTION_SPACES_FOR_TASK_AND_CELL);
    output.writeLong(task_id);
    output.writeLong(cell_id);
    output.writeObject(toByteArray(space));
    output.close();

    int iter = -1;
    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    iter = input.readInt();
    input.close();
    return iter;
  }


  public boolean deleteSolutionSpaces(long task_id) throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_DELETE_SOLUTION_SPACES_FOR_TASK);
    output.writeLong(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    boolean done = input.readBoolean();
    input.close();

    return done;
  }


  public Long[] getTaskIDs() throws IOException {
    Long[] task_ids = null;

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_TASK_IDS);

    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int n = input.readInt();

    if (n > 0) {
      task_ids = new Long[n];
      for (int i = 0; i < n; i++) {
        task_ids[i] = input.readLong();
      }
    }
    input.close();

    return task_ids;

  }


  public int getVersionOfNewSolutonSpace(long task_id, long cell_id)
      throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_NEW_SPACE_VERSION_NUMBER);
    output.writeLong(task_id);
    output.writeLong(cell_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int count = input.readInt();
    input.close();

    return count;
  }


  public EvTaskInfo getTaskForSystem(int id) throws IOException {
    EvTaskInfo info = null;
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_TASK);
    output.writeInt(id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    try {
      if (input.readBoolean()) {

        info = new EvTaskInfo();
        info.setId(input.readInt());
        info.setStatus(input.readInt());
        info.setDescription((String) input.readObject());
        info.setSubmissionTime((Timestamp) input.readObject());
        info.setJar(input.readObject());
      }
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
    input.close();
    return info;
  }


  public int addTaskForSystem(byte[] file, String desc) throws IOException {
    int ID = -1;

    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_ADD_TASK);
    output.writeObject(desc);
    output.writeObject(file);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    ID = input.readInt();
    input.close();

    return ID;
  }


  public void deleteTaskFromSystem(int id) throws IOException {

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_DELETE_TASK_FROM_SYSTEM);
    output.writeInt(id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();

  }


  public Integer[] getTaskIDsForSystem() throws IOException {

    Integer[] task_ids = null;

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_TASK_FOR_SYSTEM_IDS);

    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int n = input.readInt();

    if (n > 0) {
      task_ids = new Integer[n];
      for (int i = 0; i < n; i++) {
        task_ids[i] = input.readInt();
      }
    }
    input.close();

    return task_ids;
  }


  public void changeTaskState(int id, int state) throws IOException {

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_CHANGE_TASK_STATE_FOR_SYSTEM);
    output.writeInt(id);
    output.writeInt(state);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();
  }


  public void deleteResource(String name) throws IOException {

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_DEL_RESOURCE);
    output.writeObject(name);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();

  }


  public Object getResource(String name) throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_RESOURCE);
    output.writeObject(name);
    output.close();
    Object res = null;

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());

    if (!input.readBoolean())
      return null;

    try {
      res = fromByteArray(input.readObject());
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
      System.out.println("----");
    }
    input.close();

    return res;
  }


  public String[] getResourceNames() throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_RESOURCES_NAMES);
    output.close();
    String[] names = null;

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    try {
      names = (String[]) input.readObject();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
    input.close();

    return names;
  }


  public void setResource(Object res, String name) throws IOException {

    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_SET_RESOURCE);
    output.writeObject(name);
    output.writeObject(toByteArray(res));
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();

  }


  public void deleteStatisticForTask(long task_id) throws IOException {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_DELETE_STATISTIC_FOR_TASK);
    output.writeLong(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();
  }


  public Long[] getCellIdsWithStatistics(long task_id) throws IOException {
    Long[] cell_ids = null;

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_CELL_IDS_WITH_STATISTICS);
    output.writeLong(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int n = input.readInt();

    if (n > 0) {
      cell_ids = new Long[n];
      for (int i = 0; i < n; i++) {
        cell_ids[i] = input.readLong();
      }
    }
    input.close();

    return cell_ids;

  }


  public Long[] getNodesIdsWithStatistics(long task_id, long cell_id)
      throws IOException {
    Long[] nodes_ids = null;

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_NODES_IDS_WITH_STATISTICS);
    output.writeLong(task_id);
    output.writeLong(cell_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int n = input.readInt();

    if (n > 0) {
      nodes_ids = new Long[n];
      for (int i = 0; i < n; i++) {
        nodes_ids[i] = input.readLong();
      }
    }
    input.close();

    return nodes_ids;

  }


  public Long[] getTaskIdsWithStatistics() throws IOException {
    Long[] task_ids = null;

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_TASK_IDS_WITH_STATISTICS);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int n = input.readInt();

    if (n > 0) {
      task_ids = new Long[n];
      for (int i = 0; i < n; i++) {
        task_ids[i] = input.readLong();
      }
    }
    input.close();

    return task_ids;

  }


  public EvStatistic[] getStatistics(long task_id, long cell_id, long node_id)
      throws IOException {

    EvStatistic[] result = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output.writeInt(EvDBOperations.CMD_GET_STATISTICS);
      output.writeLong(task_id);
      output.writeLong(cell_id);
      output.writeLong(node_id);
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      int m = input.readInt();
      result = new EvStatistic[m];
      for (int i = 0; i < m; i++) {
        result[i] = (EvStatistic) (fromByteArray(input.readObject()));
      }
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace(System.out);
    }

    return result;

  }


  /**
   * Return statistics of given task of specified cells ordered by iteration
   * number
   */
  public EvStatistic[] getStatisticsByCells(long task_id, int[] cell_list)
      throws IOException {
    EvStatistic[] result = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output.writeInt(EvDBOperations.CMD_GET_STATISTICS_BY_CELLS);
      output.writeLong(task_id);
      output.writeInt(cell_list.length);
      for (int i = 0; i < cell_list.length; i++) {
        output.writeInt(cell_list[i]);
      }
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      int m = input.readInt();
      result = new EvStatistic[m];
      for (int i = 0; i < m; i++) {
        result[i] = (EvStatistic) (fromByteArray(input.readObject()));
      }
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace(System.out);
    }

    return null;
  }


  public EvStatistic[] getStatisticsByNodes(long task_id, int[] nodes)
      throws IOException {
    EvStatistic[] result = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output.writeInt(EvDBOperations.CMD_GET_STATISTICS_BY_NODES);
      output.writeLong(task_id);
      output.writeInt(nodes.length);
      for (int i = 0; i < nodes.length; i++) {
        output.writeInt(nodes[i]);
      }
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      int m = input.readInt();
      result = new EvStatistic[m];
      for (int i = 0; i < m; i++) {
        result[i] = (EvStatistic) (fromByteArray(input.readObject()));
      }
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace(System.out);
    }

    return null;
  }


  public EvStatistic[] getStatisticsByTask(long task_id) throws IOException {
    EvStatistic[] result = null;

    try {
      URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
      URLConnection connection = servlet.openConnection();
      connection.setDoOutput(true);
      ObjectOutputStream output =
          new ObjectOutputStream(connection.getOutputStream());

      output.writeInt(EvDBOperations.CMD_GET_STATISTICS_BY_ID);
      output.writeLong(task_id);
      output.close();

      ObjectInputStream input =
          new ObjectInputStream(connection.getInputStream());
      int m = input.readInt();
      result = new EvStatistic[m];
      for (int i = 0; i < m; i++) {
        result[i] = (EvStatistic) (fromByteArray(input.readObject()));
      }
      input.close();
    } catch (ClassNotFoundException e) {
      e.printStackTrace(System.out);
    }

    return null;
  }


  public void saveStatistic(long task_id, long cell_id, long node_id,
      Object stat, int iteration) throws IOException {

    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_SAVE_STATISTICS);
    output.writeLong(task_id);
    output.writeLong(cell_id);
    output.writeLong(node_id);
    output.writeInt(iteration);
    output.writeObject(toByteArray(stat));
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();
  }


  public void addIndividuals(Object[] individuals, long task_id,
      double[] objective_values, long cell_id, long node_id) throws IOException {

    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_ADD_INDIVIDUALS);
    output.writeLong(task_id);
    output.writeLong(cell_id);
    output.writeLong(node_id);
    output.writeInt(individuals.length);

    for (int i = 0; i < individuals.length; i++) {
      output.writeDouble(objective_values[i]);
      output.writeObject(toByteArray(individuals[i]));
    }
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();
  }


  /** ************************************************************* */
  // note that we set objective function to null when we send it to database (we
  // set it back fter its send)
  @SuppressWarnings("unchecked")
  public int[] addIndividualsToEval(int task_id, long cell_id, long node_id,
      int iter, Object[] individuals) throws Exception {
    List<Integer> result = new ArrayList<Integer>();

    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_ADD_INDIVIDUALS_TO_EVAL);
    output.writeInt(task_id);
    output.writeLong(cell_id);
    output.writeLong(node_id);
    output.writeInt(iter);
    output.writeInt(individuals.length);

    EvObjectiveFunction<EvIndividual> function = null;
    if (individuals != null)
      function = ((EvIndividual) individuals[0]).getObjectiveFunction();

    for (int i = 0; i < individuals.length; i++) {

      EvIndividual individual = (EvIndividual) individuals[i];
      individual.setObjectiveFunction(null);
      output.writeObject(toByteArray(individual));
      individual.setObjectiveFunction(function);
    }

    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int len = input.readInt();

    for (int i = 0; i < len; i++) {
      result.add(input.readInt());
    }

    input.close();

    int[] res = new int[result.size()];
    for (int i = 0; i < result.size(); i++) {
      res[i] = result.get(i);
    }

    return res;
  }


  public void addIndividualsValues(int[] ids, double[] values) throws Exception {
    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_ADD_INDIVIDUALS_VALUES);
    output.writeInt(ids.length);

    for (int i = 0; i < ids.length; i++) {
      output.writeInt(ids[i]);
      output.writeDouble(values[i]);
    }

    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();
  }


  public Object[] getIndividualsToEval(int[] ids) throws Exception {

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_INDIVIDUALS_TO_EVAL);
    output.writeInt(ids.length);

    for (int i = 0; i < ids.length; i++) {
      output.writeInt(ids[i]);
    }

    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int len = input.readInt();

    Object[] result = new Object[len];

    for (int i = 0; i < len; i++) {
      result[i] = fromByteArray(input.readObject());
    }

    input.close();
    return result;
  }


  public double[] getValues(int[] ids) throws Exception {
    List<Double> result = new ArrayList<Double>();

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_VALUES);
    output.writeInt(ids.length);
    for (int i = 0; i < ids.length; i++) {
      output.writeInt(ids[i]);
    }
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int len = input.readInt();

    for (int i = 0; i < len; i++) {
      result.add(input.readDouble());
    }

    input.close();

    double[] res = new double[result.size()];
    for (int i = 0; i < result.size(); i++) {
      res[i] = result.get(i);
    }

    return res;
  }


  public void deleteIndividualsToEval(int task_id) throws Exception {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_DELETE_EVALUATED_INDIVIDUALS);
    output.writeInt(task_id);

    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());

    input.close();
  }


  public Object[] getIndividualsToEvalByIteration(int task_id, long cell_id,
      long node_id, int iteration) throws Exception {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_INDIVIDUALS_TO_EVAL_BY_ITER);
    output.writeInt(task_id);
    output.writeLong(cell_id);
    output.writeLong(node_id);
    output.writeInt(iteration);

    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int len = input.readInt();

    Object[] ind_tab = new Object[len];

    for (int i = 0; i < len; i++) {
      ind_tab[i] = fromByteArray(input.readObject());
    }

    input.close();

    return ind_tab;
  }


  public void addFun(int task_id, Object fun) throws Exception {
    URL servlet = new URL(UPLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_ADD_FUN);
    output.writeInt(task_id);
    output.writeObject(toByteArray(fun));
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();

  }


  public Object getFun(int task_id) throws Exception {

    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_GET_FUN);
    output.writeInt(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    Object o = fromByteArray(input.readObject());
    input.close();
    return o;
  }


  public boolean presentFun(int task_id) throws Exception {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_CONTAIN_FUN);
    output.writeInt(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    Boolean b = input.readBoolean();
    input.close();
    return b;
  }


  public void deleteFun(int task_id) throws Exception {
    URL servlet = new URL(DOWNLOAD_SERVLET_ADDRESS);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvDBOperations.CMD_DELETE_FUN);
    output.writeInt(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.close();
  }

}
