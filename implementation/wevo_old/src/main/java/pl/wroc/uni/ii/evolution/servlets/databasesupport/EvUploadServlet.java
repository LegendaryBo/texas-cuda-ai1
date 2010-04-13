package pl.wroc.uni.ii.evolution.servlets.databasesupport;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
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

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.providers.EvDatabaseProvider;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.providers.EvDatabaseProviderFactory;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.pooling.*;

/**
 * This servlet is reponsible for:
 * <ul>
 * <li> uploading JAR files with tasks </li>
 * <li> uploading individuals </li>
 * </ul>
 * 
 * @author Marcin Golebiowski, Piotr Lipinski
 */
public class EvUploadServlet extends HttpServlet {

  private static final long serialVersionUID = 1090590698352987769L;

  private String database_url = null;

  private String database_user = null;

  private String database_password = null;

  private String driver = null;

  private String provider = null;

  private EvDatabaseProvider database = null;

  private ConnectionPool connection_pool = null;


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

  public void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {
    ObjectInputStream input = new ObjectInputStream(request.getInputStream());

    int command = input.readInt();
    switch (command) {
      case EvDBOperations.CMD_ADD_INDIVIDUAL:
        uploadIndividual(input, request, response);
        break;
      case EvDBOperations.CMD_ADD_SOLUTION_SPACES_FOR_TASK_AND_CELL:
        uploadSolutionSpace(input, request, response);
        break;
      case EvDBOperations.CMD_ADD_TASK:
        uploadTask(input, request, response);
        break;
      case EvDBOperations.CMD_SET_RESOURCE:
        setResource(input, request, response);
        break;

      case EvDBOperations.CMD_SAVE_STATISTICS:
        saveStatistic(input, request, response);
        break;

      case EvDBOperations.CMD_ADD_INDIVIDUALS:
        uploadIndividuals(input, request, response);
        break;

      case EvDBOperations.CMD_ADD_INDIVIDUALS_TO_EVAL:
        uploadIndividualsToEval(input, request, response);
        break;

      case EvDBOperations.CMD_ADD_INDIVIDUALS_VALUES:
        uploadIndividualsToEvalValues(input, request, response);
        break;

      case EvDBOperations.CMD_ADD_FUN:
        uploadFun(input, request, response);
        break;

    }

    input.close();
  }


  private void uploadFun(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) {

    try {
      int task_id = input.readInt();
      Object o = input.readObject();

      Connection conn = connection_pool.getConnection();
      database.addFun(conn, task_id, o);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();
    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }

  }


  private void uploadIndividualsToEvalValues(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {

      int ind_count = input.readInt();

      double[] values = new double[ind_count];
      int[] ids = new int[ind_count];

      for (int i = 0; i < ind_count; i++) {
        ids[i] = input.readInt();
        values[i] = input.readDouble();
      }

      Connection conn = connection_pool.getConnection();
      database.addIndividualsValues(conn, ids, values);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }

  }


  private void uploadIndividualsToEval(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {

    try {
      int task_id = input.readInt();
      long cell_id = input.readLong();
      long node_id = input.readLong();
      int iteration = input.readInt();

      int ind_count = input.readInt();
      Object[] individuals = new Object[ind_count];
      for (int i = 0; i < ind_count; i++) {
        individuals[i] = input.readObject();
      }

      Connection conn = connection_pool.getConnection();
      int[] ids =
          database.addIndividualsToEval(conn, task_id, cell_id, node_id,
              iteration, individuals);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      output.writeInt(ids.length);
      for (int i = 0; i < ids.length; i++) {
        output.writeInt(ids[i]);
      }
      output.close();

    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }
  }


  private void uploadIndividuals(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {

    try {
      long instanceID = input.readLong();
      long creationCell = input.readLong();
      long creationNode = input.readLong();
      int ind_count = input.readInt();

      double[] values = new double[ind_count];
      Object[] individuals = new Object[ind_count];

      for (int i = 0; i < ind_count; i++) {
        values[i] = input.readDouble();
        individuals[i] = input.readObject();
      }

      Connection conn = connection_pool.getConnection();
      database.addIndividuals(conn, individuals, instanceID, values,
          creationCell, creationNode);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }
  }


  private void saveStatistic(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response) {
    try {
      long task_id = input.readLong();
      long cell_id = input.readLong();
      long node_id = input.readLong();
      int iteration = input.readInt();
      Object stat = input.readObject();

      EvStatistic ev_stat = (EvStatistic) (fromByteArray(stat));
      ev_stat.setTime((new java.util.Date()).getTime());
      stat = toByteArray(ev_stat);

      Connection conn = connection_pool.getConnection();
      database.saveStatistic(conn, task_id, cell_id, node_id, stat, iteration);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"setResource\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }

  }


  private void setResource(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) {

    try {
      String name = (String) input.readObject();
      Object res = input.readObject();

      Connection conn = connection_pool.getConnection();
      database.setResource(conn, res, name);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"setResource\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.close();

    } catch (Exception ex) {
      ex.printStackTrace(System.out);
    }
  }


  private void uploadTask(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    try {
      String desc = (String) input.readObject();

      Connection conn = connection_pool.getConnection();
      int id =
          database.addTaskForSystem(conn, (byte[]) input.readObject(), desc);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(id);
      output.close();

    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (SQLException e) {
      e.printStackTrace();
    }

  }


  private void uploadSolutionSpace(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    try {
      long task_id = input.readLong();
      long cell_id = input.readLong();
      Object space = input.readObject();

      Connection conn = connection_pool.getConnection();
      int iteration = database.addSolutionSpace(conn, task_id, cell_id, space);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(iteration);
      output.close();

    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }


  private void uploadIndividual(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {
    try {

      long instanceID = input.readLong();
      double objectiveValue = input.readDouble();
      Object individual = input.readObject();
      long creationCell = input.readLong();
      long creationNode = input.readLong();

      Connection conn = connection_pool.getConnection();
      int ID =
          database.addIndividual(conn, individual, instanceID, objectiveValue,
              creationCell, creationNode);
      connection_pool.free(conn);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"evolution\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(ID);
      output.close();

    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    } catch (SQLException ex) {
      ex.printStackTrace();
    }
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
  public void destroy() {
    connection_pool.closeAllConnections();
  }
}
