package pl.wroc.uni.ii.evolution.servlets.databasesupport.providers;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.sql.Blob;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvIndividualInfo;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

public class EvMySQLDatabaseProvider implements EvDatabaseProvider {

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


  public int addIndividual(Connection connection, Object individual,
      long task_id, double objective_value, long cell_id, long node_id)
      throws IOException {
    int ID = -1;

    try {

      PreparedStatement stmt =
          connection
              .prepareStatement(
                  "INSERT INTO exchange(task_id, objective_value, individual, cell_id, node_id) VALUES("
                      + task_id
                      + ","
                      + objective_value
                      + ",?,"
                      + cell_id
                      + ","
                      + node_id + ")", Statement.RETURN_GENERATED_KEYS);

      stmt.setBytes(1, toByteArray(individual));
      stmt.execute();
      ResultSet rs = stmt.getGeneratedKeys();

      if (rs.next()) {

        ID = rs.getInt(1);
      }

      stmt.close();
      rs.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
    return ID;
  }


  public int[] addIndividuals(Connection connection, Object[] individuals,
      long task_id, double[] objective_value, long cell_id, long node_id)
      throws IOException {

    List<Integer> ids = new ArrayList<Integer>();

    try {
      PreparedStatement stmt =
          connection
              .prepareStatement(
                  "INSERT INTO exchange(task_id, objective_value, individual, cell_id, node_id) VALUES("
                      + task_id
                      + ","
                      + "?,"
                      + "?,"
                      + cell_id
                      + ","
                      + node_id
                      + ")", Statement.RETURN_GENERATED_KEYS);

      for (int i = 0; i < individuals.length; i++) {
        stmt.setDouble(1, objective_value[i]);
        stmt.setBytes(2, toByteArray(individuals[i]));
        stmt.execute();
        ResultSet rs = stmt.getGeneratedKeys();

        if (rs.next()) {
          ids.add(rs.getInt(1));
        }
        rs.close();
      }

      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }

    int[] result = new int[ids.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = ids.get(i);
    }
    return result;
  }


  /** *************************************************************************** */

  public boolean deleteIndividual(Connection connection, int id)
      throws IOException {
    boolean done = false;
    try {
      Statement stmt = connection.createStatement();
      String s = "DELETE FROM exchange WHERE id = " + id;
      stmt.execute(s);
      stmt.close();

      done = true;
    } catch (SQLException e) {
      e.printStackTrace();
    }
    return done;
  }


  /** *************************************************************************** */

  public EvIndividualInfo getIndividualInfo(Connection connection, int id,
      boolean with_individual_itself) throws IOException {
    EvIndividualInfo info = null;

    try {
      Statement stmt = connection.createStatement();
      String s = "SELECT * FROM exchange WHERE id = " + id;
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next()) {
        info = new EvIndividualInfo();
        info.setID(rs.getInt("id"));
        info.setTaskID(rs.getLong("task_id"));
        info.setObjectiveValue(rs.getDouble("objective_value"));
        info.setCellID(rs.getLong("cell_id"));
        info.setNodeID(rs.getLong("node_id"));
        info.setCreationTime(rs.getTimestamp("creation_time"));

        if (with_individual_itself) {
          Blob blob = rs.getBlob("individual");
          ObjectInputStream input =
              new ObjectInputStream(blob.getBinaryStream());
          info.setIndividual(input.readObject());
        }
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (ClassNotFoundException e) {
      e.printStackTrace(System.out);
    }

    catch (Exception ex) {
      ex.printStackTrace(System.out);
    }

    return info;
  }


  /** *************************************************************************** */

  public EvIndividualInfo[] getBestIndividualInfos(Connection connection,
      long task_id, int k, int n, boolean with_individuals_itselves)
      throws IOException {
    EvIndividualInfo[] infos = new EvIndividualInfo[n];

    try {
      int i = 0;

      Statement stmt = connection.createStatement();
      String s =
          "SELECT * FROM (SELECT * FROM exchange WHERE task_id = " + task_id
              + " ORDER BY objective_value DESC) AS cos LIMIT " + (k - 1)
              + " ," + n;
      ResultSet rs = stmt.executeQuery(s);

      while (rs.next()) {
        infos[i] = new EvIndividualInfo();
        infos[i].setID(rs.getInt("id"));
        infos[i].setTaskID(rs.getLong("task_id"));
        infos[i].setObjectiveValue(rs.getDouble("objective_value"));
        infos[i].setCellID(rs.getLong("cell_id"));
        infos[i].setNodeID(rs.getLong("node_id"));
        infos[i].setCreationTime(rs.getTimestamp("creation_time"));

        if (with_individuals_itselves) {
          Blob blob = rs.getBlob("individual");
          ObjectInputStream input =
              new ObjectInputStream(blob.getBinaryStream());
          infos[i].setIndividual(input.readObject());
        }

        i++;
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return infos;
  }


  public EvIndividualInfo[] getBestIndividualInfosMatchingCell(
      Connection connection, long task_id, long cell_id, int k, int n,
      boolean with_individuals_itselves) throws IOException {
    EvIndividualInfo[] infos = new EvIndividualInfo[n];

    try {
      int i = 0;

      Statement stmt = connection.createStatement();
      String s =
          "SELECT * FROM (SELECT * FROM exchange WHERE task_id = " + task_id
              + " AND cell_id = " + cell_id
              + " ORDER BY objective_value DESC) AS cos LIMIT " + (k - 1)
              + " ," + n;
      ResultSet rs = stmt.executeQuery(s);

      while (rs.next()) {
        infos[i] = new EvIndividualInfo();
        infos[i].setID(rs.getInt("id"));
        infos[i].setTaskID(rs.getLong("task_id"));
        infos[i].setObjectiveValue(rs.getDouble("objective_value"));
        infos[i].setCellID(rs.getLong("cell_id"));
        infos[i].setNodeID(rs.getLong("node_id"));
        infos[i].setCreationTime(rs.getTimestamp("creation_time"));

        if (with_individuals_itselves) {
          Blob blob = rs.getBlob("individual");
          ObjectInputStream input =
              new ObjectInputStream(blob.getBinaryStream());
          infos[i].setIndividual(input.readObject());
        }

        i++;
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return infos;
  }


  public EvIndividualInfo[] getBestIndividualInfosNotMatchingCell(
      Connection connection, long task_id, long cell_id, int k, int n,
      boolean with_individuals_itselves) throws IOException {
    EvIndividualInfo[] infos = new EvIndividualInfo[n];

    try {
      int i = 0;

      Statement stmt = connection.createStatement();
      String s =
          "SELECT * FROM (SELECT * FROM exchange WHERE task_id = " + task_id
              + " AND cell_id <> " + cell_id
              + " ORDER BY objective_value DESC) AS cos LIMIT " + (k - 1)
              + " ," + n;
      ResultSet rs = stmt.executeQuery(s);

      while (rs.next()) {
        infos[i] = new EvIndividualInfo();
        infos[i].setID(rs.getInt("id"));
        infos[i].setTaskID(rs.getLong("task_id"));
        infos[i].setObjectiveValue(rs.getDouble("objective_value"));
        infos[i].setCellID(rs.getLong("cell_id"));
        infos[i].setNodeID(rs.getLong("node_id"));
        infos[i].setCreationTime(rs.getTimestamp("creation_time"));

        if (with_individuals_itselves) {
          Blob blob = rs.getBlob("individual");
          ObjectInputStream input =
              new ObjectInputStream(blob.getBinaryStream());
          infos[i].setIndividual(input.readObject());
        }

        i++;
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return infos;
  }


  public EvIndividualInfo[] getBestIndividualInfosMatchingNode(
      Connection connection, long task_id, long cell_id, int k, int n,
      boolean with_individuals_itselves) throws IOException {
    EvIndividualInfo[] infos = new EvIndividualInfo[n];

    try {
      int i = 0;

      Statement stmt = connection.createStatement();
      String s =
          "SELECT * FROM (SELECT * FROM exchange WHERE task_id = " + task_id
              + " AND node_id = " + cell_id
              + " ORDER BY objective_value DESC) AS Cos LIMIT " + (k - 1)
              + " ," + n;
      ResultSet rs = stmt.executeQuery(s);

      while (rs.next()) {
        infos[i] = new EvIndividualInfo();
        infos[i].setID(rs.getInt("id"));
        infos[i].setTaskID(rs.getLong("task_id"));
        infos[i].setObjectiveValue(rs.getDouble("objective_value"));
        infos[i].setCellID(rs.getLong("cell_id"));
        infos[i].setNodeID(rs.getLong("node_id"));
        infos[i].setCreationTime(rs.getTimestamp("creation_time"));

        if (with_individuals_itselves) {
          Blob blob = rs.getBlob("individual");
          ObjectInputStream input =
              new ObjectInputStream(blob.getBinaryStream());
          infos[i].setIndividual(input.readObject());
        }

        i++;
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return infos;
  }


  /** *************************************************************************** */

  public int getIndividualCount(Connection connection, long task_id)
      throws IOException {
    int count = -1;

    try {
      Statement stmt = connection.createStatement();
      String s = "SELECT COUNT(*) FROM exchange WHERE task_id = " + task_id;
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next())
        count = rs.getInt(1);

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }

    return count;
  }


  public int getIndividualCountMatchingCell(Connection connection,
      long task_id, long cell_id) throws IOException {
    int count = -1;

    try {
      Statement stmt = connection.createStatement();
      String s =
          "SELECT COUNT(*) FROM exchange WHERE task_id = " + task_id
              + "AND cell_id = " + cell_id;
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next())
        count = rs.getInt(1);

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }

    return count;
  }


  public int getIndividualCountNotMatchingCell(Connection connection,
      long task_id, long cell_id) throws IOException {
    int count = -1;

    try {

      Statement stmt = connection.createStatement();
      String s =
          "SELECT COUNT(*) FROM exchange WHERE task_id = " + task_id
              + "AND cell_id <> " + cell_id;
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next())
        count = rs.getInt(1);

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }

    return count;
  }


  public int getIndividualCountMatchingNode(Connection connection,
      long task_id, long cell_id) throws IOException {
    int count = -1;

    try {

      Statement stmt = connection.createStatement();
      String s =
          "SELECT COUNT(*) FROM exchange WHERE task_id = " + task_id
              + "AND node_id = " + cell_id;
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next())
        count = rs.getInt(1);

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }

    return count;
  }


  public void deleteIndividualsFromTask(Connection connection, long task_id)
      throws IOException {
    try {
      Statement stmt = connection.createStatement();
      String s = "DELETE FROM exchange WHERE task_id=" + task_id;
      stmt.execute(s);
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  public int addSolutionSpace(Connection connection, long task_id,
      long cell_id, Object space) throws IOException {
    int ID = -1;

    try {

      PreparedStatement stmt =
          connection
              .prepareStatement(
                  "INSERT INTO solutionspaces(task_id,  cell_id, solution_space) VALUES(?, ?, ?)",
                  Statement.RETURN_GENERATED_KEYS);

      stmt.setLong(1, task_id);
      stmt.setLong(2, cell_id);
      stmt.setBytes(3, toByteArray(space));
      stmt.execute();

      ResultSet rs = stmt.getGeneratedKeys();

      if (rs.next()) {

        ID = rs.getInt(1);
      }
      rs.close();
      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
    return ID;

  }


  public Object getSolutionSpace(Connection connection, long task_id,
      long cell_id) throws IOException {

    Object space = null;

    try {

      Statement stmt = connection.createStatement();
      String s =
          "SELECT solution_space FROM solutionspaces WHERE iteration = (SELECT MAX(iteration) FROM solutionspaces WHERE task_id="
              + task_id + " AND cell_id=" + cell_id + ")";
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next()) {
        Blob blob = rs.getBlob(1);
        ObjectInputStream input = new ObjectInputStream(blob.getBinaryStream());
        space = input.readObject();
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }

    return space;
  }


  public boolean deleteSolutionSpaces(Connection connection, long task_id)
      throws IOException {
    try {
      Statement stmt = connection.createStatement();
      String s = "DELETE FROM solutionspaces WHERE task_id=" + task_id;
      stmt.execute(s);
      stmt.close();
      return true;
    } catch (SQLException e) {
      e.printStackTrace();
    }

    return false;
  }


  public Long[] getTaskIDs(Connection connection) {
    ArrayList<Long> list = new ArrayList<Long>();

    try {
      Statement stmt = connection.createStatement();
      String s = "SELECT DISTINCT task_id FROM exchange";
      ResultSet rs = stmt.executeQuery(s);

      while (rs.next()) {
        list.add(rs.getLong(1));
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }

    if (list.size() != 0) {
      Long[] res = new Long[list.size()];

      for (int i = 0; i < list.size(); i++) {
        res[i] = list.get(i);
      }

      return res;
    }

    return null;
  }


  public int getVersionOfNewSolutonSpace(Connection connection, long task_id,
      long cell_id) {
    int count = 0;

    try {

      Statement stmt = connection.createStatement();
      String s =
          "SELECT MAX(iteration) FROM solutionspaces WHERE task_id=" + task_id
              + " AND cell_id=" + cell_id;
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next()) {
        count = rs.getInt(1);
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }

    return count;
  }


  public EvTaskInfo getTaskForSystem(Connection connection, int id)
      throws IOException {
    EvTaskInfo info = null;

    try {
      Statement stmt = connection.createStatement();
      String s = "SELECT * FROM tasks WHERE id=" + id;
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next()) {

        Blob blob = rs.getBlob("jar");
        byte[] o = new byte[(int) blob.length()];
        blob.getBinaryStream().read(o);
        info = new EvTaskInfo();
        info.setId(rs.getInt("id"));
        info.setJar(o);
        info.setStatus(rs.getInt("status"));
        info.setDescription(rs.getString("description"));
        info.setSubmissionTime(rs.getTimestamp("submission_time"));

      }
      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }
    return info;
  }


  public int addTaskForSystem(Connection connection, byte[] file, String desc)
      throws IOException {

    int ID = -1;

    try {

      PreparedStatement stmt =
          connection.prepareStatement(
              "INSERT INTO tasks(jar, status, description) VALUES(?, 1, ?)",
              Statement.RETURN_GENERATED_KEYS);
      stmt.setBytes(1, file);
      stmt.setString(2, desc);
      stmt.execute();

      ResultSet rs = stmt.getGeneratedKeys();

      if (rs.next()) {

        ID = rs.getInt(1);
      }
      rs.close();
      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
    return ID;
  }


  public void deleteTaskFromSystem(Connection connection, int id)
      throws IOException {
    try {
      Statement stmt = connection.createStatement();
      String s = "DELETE FROM tasks WHERE id = " + id;
      stmt.execute(s);
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  public Integer[] getTaskIDsForSystem(Connection connection)
      throws IOException {
    ArrayList<Integer> list = new ArrayList<Integer>();

    try {
      Statement stmt = connection.createStatement();
      String s = "SELECT id FROM tasks";
      ResultSet rs = stmt.executeQuery(s);

      while (rs.next()) {
        list.add(rs.getInt(1));
      }

      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }

    if (list.size() != 0) {
      Integer[] res = new Integer[list.size()];

      for (int i = 0; i < list.size(); i++) {
        res[i] = list.get(i);
      }

      return res;
    }

    return null;
  }


  public void changeTaskState(Connection connection, int id, int state)
      throws IOException {
    try {
      Statement stmt = connection.createStatement();
      String s = "UPDATE tasks SET status=" + state + " WHERE id = " + id;
      stmt.execute(s);
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace();
    }

  }


  /** ----- */
  public boolean deleteResource(Connection connection, String name)
      throws IOException {
    boolean done = false;
    try {
      Statement stmt = connection.createStatement();
      String s = "DELETE FROM resources WHERE name ='" + name + "'";
      stmt.execute(s);
      stmt.close();
      done = true;
    } catch (SQLException e) {
    }

    return done;
  }


  public Object getResource(Connection connection, String name)
      throws IOException {
    Object res = null;

    try {
      Statement stmt = connection.createStatement();
      String s = "SELECT * FROM resources WHERE name='" + name + "'";
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next()) {

        Blob blob = rs.getBlob("resource");

        ObjectInputStream input = new ObjectInputStream(blob.getBinaryStream());
        res = input.readObject();

      }
      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }
    return res;

  }


  public String[] getResourceNames(Connection connection) throws IOException {

    List<String> list;
    try {
      Statement stmt = connection.createStatement();
      String s = "SELECT name FROM resources";
      ResultSet rs = stmt.executeQuery(s);

      list = new ArrayList<String>();
      while (rs.next()) {
        list.add(rs.getString(1));
      }

    } catch (SQLException ex) {
      ex.printStackTrace(System.out);
      return null;
    }

    if (list.size() != 0) {
      String[] res = new String[list.size()];

      for (int i = 0; i < list.size(); i++) {
        res[i] = list.get(i);
      }
      return res;
    } else {
      return null;
    }
  }


  public boolean setResource(Connection connection, Object res, String name)
      throws IOException {
    if (deleteResource(connection, name) == false) {
      return false;
    }

    boolean done = false;

    try {

      PreparedStatement stmt =
          connection
              .prepareStatement("INSERT INTO resources(name,  resource) VALUES(?, ?)");
      stmt.setString(1, name);
      stmt.setBytes(2, toByteArray(res));
      stmt.execute();

      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace(System.out);
    }
    return done;

  }


  /** ------------------* */
  /** -- STATISTICS --* */
  /** ------------------* */
  public Long[] getCellIdsWithStatistics(Connection connection, long task_id) {
    List<Long> list;
    try {
      Statement stmt = connection.createStatement();
      String s =
          "SELECT DISTINCT cell_id FROM statistics WHERE task_id=" + task_id;
      ResultSet rs = stmt.executeQuery(s);

      list = new ArrayList<Long>();
      while (rs.next()) {
        list.add(rs.getLong(1));
      }

      rs.close();
      stmt.close();

    } catch (SQLException ex) {
      ex.printStackTrace(System.out);
      return null;
    }

    if (list.size() != 0) {
      Long[] res = new Long[list.size()];

      for (int i = 0; i < list.size(); i++) {
        res[i] = list.get(i);
      }
      return res;
    } else {
      return null;
    }

  }


  public Long[] getNodesIdsWithStatistics(Connection connection, long task_id,
      long cell_id) {
    List<Long> list;
    try {
      Statement stmt = connection.createStatement();
      String s =
          "SELECT DISTINCT node_id FROM statistics WHERE task_id='" + task_id
              + "' AND cell_id='" + cell_id + "'";
      ResultSet rs = stmt.executeQuery(s);

      list = new ArrayList<Long>();
      while (rs.next()) {
        list.add(rs.getLong(1));
      }
    } catch (SQLException ex) {
      ex.printStackTrace(System.out);
      return null;
    }

    if (list.size() != 0) {
      Long[] res = new Long[list.size()];

      for (int i = 0; i < list.size(); i++) {
        res[i] = list.get(i);
      }
      return res;
    } else {
      return null;
    }

  }


  public Long[] getTaskIdsWithStatistics(Connection connection) {
    List<Long> list;
    try {
      Statement stmt = connection.createStatement();
      String s = "SELECT DISTINCT task_id FROM statistics";
      ResultSet rs = stmt.executeQuery(s);

      list = new ArrayList<Long>();
      while (rs.next()) {
        list.add(rs.getLong(1));
      }

      rs.close();
      stmt.close();
    } catch (SQLException ex) {
      ex.printStackTrace(System.out);
      return null;
    }

    if (list.size() != 0) {
      Long[] res = new Long[list.size()];

      for (int i = 0; i < list.size(); i++) {
        res[i] = list.get(i);
      }
      return res;
    } else {
      return null;
    }
  }


  public void saveStatistic(Connection connection, long task_id, long cell_id,
      long node_id, Object stat, long iteration) {

    try {

      PreparedStatement stmt =
          connection
              .prepareStatement("INSERT INTO statistics(task_id, cell_id, node_id, iteration, ev_stat) VALUES(?, ?, ?, ?, ?)");

      stmt.setLong(1, task_id);
      stmt.setLong(2, cell_id);
      stmt.setLong(3, node_id);
      stmt.setLong(4, iteration);
      stmt.setBytes(5, toByteArray(stat));
      stmt.execute();

      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace(System.out);
    }

  }


  public void deleteStatisticForTask(Connection connection, long task_id) {
    try {
      Statement stmt = connection.createStatement();
      String s = "DELETE FROM statistics WHERE task_id = '" + task_id + "'";
      stmt.execute(s);
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    }
  }


  public Object[] getStatistics(Connection connection, long task_id,
      long cell_id, long node_id) {

    List<Object> result = new ArrayList<Object>();
    try {
      Statement stmt = connection.createStatement();
      String s =
          "SELECT * FROM statistics WHERE node_id='" + node_id
              + "' and cell_id='" + cell_id + "' and task_id='" + task_id + "'";
      ResultSet rs = stmt.executeQuery(s);
      while (rs.next()) {

        Blob blob = rs.getBlob("ev_stat");

        ObjectInputStream input = new ObjectInputStream(blob.getBinaryStream());
        result.add(input.readObject());

      }
      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }

    if (result.size() == 0) {
      return null;
    } else {
      Object[] res = new Object[result.size()];

      for (int i = 0; i < res.length; i++) {
        res[i] = result.get(i);
      }
      return res;
    }
  }


  public Object[] getStatisticsByCells(Connection connection, long task_id,
      int[] cells) {

    List<Object> result = new ArrayList<Object>();
    try {

      if (cells.length == 0)
        return null;
      String sql_str = "( cell_id=" + cells[0];
      for (int i = 1; i < cells.length; i++) {
        sql_str += " OR cell_id=" + cells[i];
      }
      sql_str += " )";

      Statement stmt = connection.createStatement();

      String s =
          "SELECT * FROM statistics WHERE " + "task_id='" + task_id + "' AND "
              + sql_str + " ORDER BY iteration, node_id";

      ResultSet rs = stmt.executeQuery(s);
      while (rs.next()) {

        Blob blob = rs.getBlob("ev_stat");

        ObjectInputStream input = new ObjectInputStream(blob.getBinaryStream());
        result.add((EvStatistic) input.readObject());

      }
      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }

    if (result.size() == 0) {
      return null;
    } else {
      Object[] res = new Object[result.size()];

      for (int i = 0; i < res.length; i++) {
        res[i] = result.get(i);
      }
      return res;
    }
  }


  public EvStatistic[] getStatisticsByNodes(Connection connection,
      long task_id, int[] nodes) {

    List<EvStatistic> result = new ArrayList<EvStatistic>();
    try {

      if (nodes.length == 0)
        return null;
      String sql_str = "( node_id=" + nodes[0];
      for (int i = 1; i < nodes.length; i++) {
        sql_str += " OR nodes_id=" + nodes[i];
      }
      sql_str += " )";

      Statement stmt = connection.createStatement();

      String s =
          "SELECT * FROM statistics WHERE " + "task_id='" + task_id + "' AND "
              + sql_str + " ORDER BY iteration, node_id";

      ResultSet rs = stmt.executeQuery(s);
      while (rs.next()) {

        Blob blob = rs.getBlob("ev_stat");

        ObjectInputStream input = new ObjectInputStream(blob.getBinaryStream());
        result.add((EvStatistic) input.readObject());

      }
      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }

    if (result.size() == 0) {
      return null;
    } else {
      EvStatistic[] res = new EvStatistic[result.size()];

      for (int i = 0; i < res.length; i++) {
        res[i] = result.get(i);
      }
      return res;
    }
  }


  public EvStatistic[] getStatisticsByID(Connection connection, long task_id) {
    List<EvStatistic> result = new ArrayList<EvStatistic>();
    try {
      Statement stmt = connection.createStatement();
      String s =
          "SELECT * FROM statistics WHERE " + "task_id='" + task_id
              + "' ORDER BY iteration, node_id";
      ResultSet rs = stmt.executeQuery(s);
      while (rs.next()) {

        Blob blob = rs.getBlob("ev_stat");

        ObjectInputStream input = new ObjectInputStream(blob.getBinaryStream());
        result.add((EvStatistic) input.readObject());

      }
      rs.close();
      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }

    if (result.size() == 0) {
      return null;
    } else {
      EvStatistic[] res = new EvStatistic[result.size()];

      for (int i = 0; i < res.length; i++) {
        res[i] = result.get(i);
      }
      return res;
    }
  }


  public void init() {
    try {
      DriverManager.registerDriver(new com.mysql.jdbc.Driver());
    } catch (SQLException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }


  public int[] addIndividualsToEval(Connection connection, int task_id,
      long cell_id, long node_id, int iteration, Object[] individuals)
      throws Exception {
    List<Integer> ids = new ArrayList<Integer>();

    try {
      PreparedStatement stmt =
          connection
              .prepareStatement(
                  "INSERT INTO eval_individuals(task_id, individual, cell_id, node_id, iteration) VALUES("
                      + task_id
                      + ","
                      + "?,"
                      + cell_id
                      + ","
                      + node_id
                      + ","
                      + iteration + ")", Statement.RETURN_GENERATED_KEYS);

      for (int i = 0; i < individuals.length; i++) {
        stmt.setBytes(1, toByteArray(individuals[i]));
        stmt.execute();
        ResultSet rs = stmt.getGeneratedKeys();

        if (rs.next()) {
          ids.add(rs.getInt(1));
        }
        rs.close();
      }

      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }

    int[] result = new int[ids.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = ids.get(i);
    }
    return result;
  }


  public void addIndividualsValues(Connection connection, int[] ids,
      double[] values) {
    long start_s = System.currentTimeMillis();
    try {
      PreparedStatement stmt =
          connection.prepareStatement("INSERT INTO eval_values VALUES(?, ?)");
      for (int i = 0; i < ids.length; i++) {
        try {
          stmt.setInt(1, ids[i]);
          stmt.setDouble(2, values[i]);
          stmt.execute();
        } catch (SQLException e) {
        }
      }

      stmt.close();
    } catch (Exception e) {
      return;
    }
    long end_ev = System.currentTimeMillis();
    System.out.println("UPLOAD TOOK" + (int) (end_ev - start_s) + "ms");
  }


  public Object[] getIndividualsToEval(Connection connection, int[] ids) {

    List<Object> result = new ArrayList<Object>();
    try {
      PreparedStatement stmt =
          connection
              .prepareStatement("SELECT * FROM eval_individuals WHERE id=?");

      for (int i = 0; i < ids.length; i++) {
        stmt.setInt(1, ids[i]);
        stmt.execute();

        ResultSet rs = stmt.getResultSet();

        if (rs.next()) {
          Blob blob = rs.getBlob("individual");
          ObjectInputStream input =
              new ObjectInputStream(blob.getBinaryStream());
          result.add(input.readObject());
        }
        rs.close();
      }

      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }

    if (result.size() == 0) {
      return null;
    } else {
      Object[] res = new Object[result.size()];

      for (int i = 0; i < res.length; i++) {
        res[i] = result.get(i);
      }
      return res;
    }
  }


  public double[] getValues(Connection connection, int[] ids) {
    List<Double> values = new ArrayList<Double>();

    try {
      PreparedStatement stmt =
          connection
              .prepareStatement("SELECT ind_value  FROM eval_values WHERE id = ?");

      for (int i = 0; i < ids.length; i++) {
        stmt.setInt(1, ids[i]);
        stmt.execute();
        ResultSet rs = stmt.getResultSet();

        if (rs.next()) {
          values.add(rs.getDouble(1));
        }
        rs.close();
      }

      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }

    double[] result = new double[values.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = values.get(i);
    }
    return result;
  }


  public void deleteEvaledIndividuals(Connection connection, int task_id) {
    PreparedStatement stmt;
    try {

      stmt =
          connection
              .prepareStatement("DELETE FROM eval_individuals WHERE task_id=?");
      stmt.setInt(1, task_id);
      stmt.execute();
      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  public Object[] getIndividualsToEvalByIter(Connection connection,
      int task_id, long cell_id, long node_id, int iteration) {

    List<Object> result = new ArrayList<Object>();
    try {
      PreparedStatement stmt =
          connection
              .prepareStatement("SELECT * FROM eval_individuals WHERE task_id=? AND cell_id=? AND node_id=? AND iteration=?");
      stmt.setInt(1, task_id);
      stmt.setLong(2, cell_id);
      stmt.setLong(3, node_id);
      stmt.setInt(4, iteration);
      stmt.execute();

      ResultSet rs = stmt.getResultSet();
      while (rs.next()) {
        Blob blob = rs.getBlob("individual");
        ObjectInputStream input = new ObjectInputStream(blob.getBinaryStream());
        result.add(input.readObject());
      }
      rs.close();

      stmt.close();
    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }

    if (result.size() == 0) {
      return null;
    } else {
      Object[] res = new Object[result.size()];

      for (int i = 0; i < res.length; i++) {
        res[i] = result.get(i);
      }
      return res;
    }
  }


  public void addFun(Connection connection, int task_id, Object fun) {
    try {
      PreparedStatement stmt =
          connection
              .prepareStatement("INSERT INTO objective_function(task_id, fun) VALUES(?,?)");
      stmt.setInt(1, task_id);
      stmt.setBytes(2, toByteArray(fun));
      stmt.execute();
      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace(System.out);
    }
  }


  public Object getFun(Connection connection, int task_id) {

    Object res = null;

    try {
      Statement stmt = connection.createStatement();
      String s =
          "SELECT * FROM objective_function WHERE task_id='" + task_id + "'";
      ResultSet rs = stmt.executeQuery(s);

      if (rs.next()) {
        Blob blob = rs.getBlob("fun");
        ObjectInputStream input = new ObjectInputStream(blob.getBinaryStream());
        res = input.readObject();
      }
      rs.close();
      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }
    return res;

  }


  public boolean presentFun(Connection connection, int task_id) {
    boolean res = true;

    try {
      Statement stmt = connection.createStatement();
      String s =
          "SELECT * FROM objective_function WHERE task_id='" + task_id + "'";
      ResultSet rs = stmt.executeQuery(s);

      res = rs.next();
      rs.close();
      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }
    return res;
  }


  public void deleteFun(Connection connection, int task_id) {
    try {
      Statement stmt = connection.createStatement();
      String s =
          "DELETE FROM objective_function WHERE task_id='" + task_id + "'";
      stmt.execute(s);
      stmt.close();

    } catch (SQLException e) {
      e.printStackTrace(System.out);
    } catch (Exception e) {
      e.printStackTrace(System.out);
    }
  }

}
