package pl.wroc.uni.ii.evolution.servlets.managment.communication;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URL;
import java.net.URLConnection;
import java.sql.Timestamp;

import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

/**
 * Proxy to managmanet servlet.
 * 
 * @author Piotr Lipinski, Marcin Golebiowski
 */
public class EvManagmentServletCommunicationImpl implements
    EvManagmentServletCommunication {

  private static final int CMD_GET_NODE_ID = 101;

  private static final int CMD_GET_INSTANCE_ID = 102;

  private static final int CMD_GET_JAR_FILE = 103;

  private static final int CMD_KEEP_ALIVE = 104;

  private static final int CMD_KEEP_ALIVE_OK = 201;

  private static final int CMD_STOP_TASK = 105;

  private static final int CMD_RESUME_TASK = 106;

  private static final int CMD_CLEAR_TASK = 107;

  private static final int CMD_ADD_TASK = 108;

  private static final int CMD_DELETE_TASK = 109;

  private static final int CMD_GET_TASK_IDS = 110;

  private static final int CMD_GET_TASKINFO = 111;

  private static final int CMD_GET_NODES_COUNT = 112;

  private static final int MAX_TRIES_NUMBER = 40;

  private static final int SLEEP_TIME = 1000;

  private String managment_servlet_url = null;


  /**
   * @param managment_servlet location of management servlet (url)
   */
  public EvManagmentServletCommunicationImpl(String managment_servlet) {
    this.managment_servlet_url = managment_servlet;
  }


  /**
   * {@inheritDoc}
   */
  public void deleteTask(int task_id) throws Exception {
    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_DELETE_TASK);
    output.writeInt(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.readBoolean();
    input.close();
  }


  /**
   * {@inheritDoc}
   */
  public long getNodeID() throws IOException {
    // Registering on the server and getting the node ID
    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_GET_NODE_ID);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    long nodeID = input.readLong();
    input.close();

    if (nodeID == -1) {
      throw new IOException("There isn't a place for new connection in servlet");
    }
    return nodeID;
  }


  /**
   * {@inheritDoc}
   */
  public String getServletURL() {
    return this.managment_servlet_url;
  }


  /**
   * {@inheritDoc}
   */
  public int getTaskID(long node_id) throws Exception {
    // Getting the instance ID
    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_GET_INSTANCE_ID);
    output.writeLong(node_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int instanceID = input.readInt();
    input.close();

    if (instanceID == -1) {
      throw new Exception("Task list for system is empty");
    }

    return instanceID;
  }


  /**
   * {@inheritDoc}
   */
  public boolean keepAlive(long node_id) throws Exception {

    for (int tries = 0; tries < MAX_TRIES_NUMBER; tries++) {
      try {
        URL servlet = new URL(managment_servlet_url);
        URLConnection connection = servlet.openConnection();
        connection.setDoOutput(true);

        ObjectOutputStream output =
            new ObjectOutputStream(connection.getOutputStream());
        output.writeInt(CMD_KEEP_ALIVE);
        output.writeLong(node_id);
        output.close();

        ObjectInputStream input =
            new ObjectInputStream(connection.getInputStream());
        int response = input.readInt();
        input.close();
        return response == CMD_KEEP_ALIVE_OK;

      } catch (IOException e) {
      }
      waitAMoment();
    }
    throw new IOException("Problem with connection to database");

  }


  public void resumeTask(int task_id) throws Exception {

    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_RESUME_TASK);
    output.writeInt(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.readBoolean();
    input.close();

  }


  /**
   * {@inheritDoc}
   */
  public void stopTask(int task_id) throws Exception {

    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_STOP_TASK);
    output.writeInt(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.readBoolean();
    input.close();
  }


  private void waitAMoment() {
    try {
      Thread.sleep(SLEEP_TIME);
    } catch (InterruptedException e) {
      return;
    }
  }


  /**
   * {@inheritDoc}
   */
  public int addTask(byte[] file, String desc) throws Exception {

    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_ADD_TASK);
    output.writeObject(file);
    output.writeObject(desc);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int id = input.readInt();
    input.close();

    return id;

  }


  /**
   * {@inheritDoc}
   */
  public EvTaskInfo getEvTask(int task_id, boolean with_jar) throws Exception {

    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_GET_TASKINFO);
    output.writeInt(task_id);
    output.writeBoolean(with_jar);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    EvTaskInfo res = new EvTaskInfo();

    if (!input.readBoolean())
      return null;

    res.setDescription((String) input.readObject());
    res.setId((Integer) input.readObject());
    res.setStatus((Integer) input.readObject());
    res.setSubmissionTime((Timestamp) input.readObject());
    res.setJar(input.readObject());
    input.close();

    return res;

  }


  /**
   * {@inheritDoc}
   */
  public int[] getEvTaskIds() throws Exception {

    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_GET_TASK_IDS);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());

    int n = input.readInt();

    int[] tab = new int[n];

    for (int i = 0; i < n; i++) {
      tab[i] = input.readInt();
    }

    return tab;
  }


  public byte[] getJAR(long node_id) throws Exception {

    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_GET_JAR_FILE);
    output.writeLong(node_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    byte[] file = (byte[]) input.readObject();

    return file;
  }


  public int getNodeCountForTask(int task_id) throws Exception {
    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_GET_NODES_COUNT);
    output.writeInt(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int res = input.readInt();
    input.close();

    return res;
  }


  public void clearTask(int task_id) throws Exception {

    URL servlet = new URL(managment_servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);

    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(CMD_CLEAR_TASK);
    output.writeInt(task_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    input.readBoolean();
    input.close();
  }


  public String getURL() {
    return managment_servlet_url;
  }

}
