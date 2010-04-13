package pl.wroc.uni.ii.evolution.servlets.managment;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;

/**
 * Servlet responsible for distribution. It coopereats with databasesupport
 * servlets. It controlls proccess of distributed computing:
 * <ul>
 * <li> assign node </li>
 * <li> assign tasks for nodes </li>
 * <li> send task to node </li>
 * <li> manage execution of tasks</li>
 * 
 * @author Marcin Golebiowski, Piotr Lipinski
 */

public class EvDistributionController {

  private static final int CMD_KEEP_ALIVE_OK = 201;

  private static final int CMD_KEEP_ALIVE_SUICIDE = 202;

  private EvNodesManager node_manager;

  private EvTasksManager task_manager;

  private EvDBServletCommunication database_servlets;


  /**
   * @param download_servlet_url a location of servlet responsible for
   *        downloading
   * @param upload_servlet_url a loction of servlet responsible for uploading
   */
  public EvDistributionController(String wevo_url) {

    database_servlets = new EvDBServletCommunicationImpl(wevo_url);
    node_manager = new EvNodesManager();
    task_manager = new EvTasksManager(database_servlets);
  }


  /*
   * public EvDistributionController(String download_url, String upload_url) {
   * database_servlets = new EvDBServletCommunicationImpl(download_url,
   * upload_url); node_manager = new EvNodesManager(); task_manager = new
   * EvTasksManager(database_servlets); }
   */

  /**
   * Handles add task request. It reads byte array from input and task
   * desciption. It connects to a databasesupport servlet to add task to
   * database. Then it stores task in local memory. At the end it writes id of
   * new task to response stream.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   * @throws IOException
   */
  public void addTask(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {

    byte[] file = null;
    String task_description = null;

    try {

      file = (byte[]) input.readObject();
      task_description = (String) input.readObject();

    } catch (ClassNotFoundException e) {
      e.printStackTrace(System.out);
    }

    int new_task_id =
        database_servlets.addTaskForSystem(file, task_description);

    EvTaskInfo new_task = database_servlets.getTaskForSystem(new_task_id);
    task_manager.addTask(new_task);
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());
    output.writeInt(new_task_id);
    output.close();
  }


  /**
   * Handles <b> clear task </b> request. It reads task identifier from input
   * stream. It connects to a databasesupport servlet to delete all individuals
   * computed for a task. At the end it writes <code> true </code> to response
   * stream.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   * @throws IOException
   */
  public void clearTask(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    int task_id = input.readInt();
    database_servlets.deleteIndividualsFromTask(task_id);
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());
    output.writeBoolean(true);
    output.close();
  }


  /**
   * Handles <b> delete task </b> request. At the end it writes
   * <code> true </code> in response stream.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   * @throws IOException
   * @throws IOException
   */
  public void deleteTask(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    int task_id = input.readInt();
    node_manager.deleteNodesForTask(task_id);
    database_servlets.deleteTaskFromSystem(task_id);
    database_servlets.deleteIndividualsFromTask(task_id);
    task_manager.deleteTask(task_id);

    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());
    output.writeBoolean(true);
    output.close();
  }


  /**
   * Handles <b> resume task </b> request. Resumes execution of all nodes
   * working on given task. It reads task identifier from input stream. It
   * changes a state of task in local memory then it connects to a
   * databasesupport servlet in order to make this change too. At the end it
   * writes <code> true </code> to response stream.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   */
  public void resumeTask(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    int task_id = input.readInt();
    database_servlets.changeTaskState(task_id, 1);
    task_manager.changeTaskState(task_id, 1);
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());
    output.writeBoolean(true);
    output.close();
  }


  /**
   * Handles <b> send jar file </b> request. It reads a indetifier of node from
   * input stream. Then it checks if node have correct keepalive timestamp. Then
   * task manger assigns task for node and writes it to response.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   */
  public void sendJARFile(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    long node_id = input.readLong();
    if (node_manager.checkNodeID(node_id)) {
      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"Evolution.jar\"");

      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      byte[] file =
          (byte[]) task_manager.getTask(node_manager.getTaskForNode(node_id))
              .getJar();
      output.writeObject(file);
      output.flush();
      output.close();
    } else {
      response.sendError(HttpServletResponse.SC_FORBIDDEN);
    }
  }


  /**
   * Handles <b> send keep alive </b> request. Resumes execution of all nodes
   * working on given task. It reads task identifier from input stream. It
   * changes a state of task in local memory then it connects to a
   * databasesupport servlet in order to make this change too. At the end it
   * writes <code> true </code> in response stream.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   */
  public void sendKeepAlive(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    long nodeID = input.readLong();

    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"KeepAliveOK\"");
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());

    node_manager.checkTimeouts();

    if (node_manager.checkNodeID(nodeID)) {
      output.writeInt(CMD_KEEP_ALIVE_OK);
    } else {
      output.writeInt(CMD_KEEP_ALIVE_SUICIDE);
    }
    output.close();
  }


  /**
   * Handles <b> send task id </b> request. It reads a indetifier of node from
   * input stream.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   */
  public void sendNodeID(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {

    long node_id = node_manager.assignNodeID();
    response.setContentType("application/octet-stream");
    response
        .setHeader("content-disposition", "attachment; filename=\"NodeID\"");
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());
    output.writeLong(node_id);
    output.close();

  }


  /**
   * Handles <b> send task id </b> request. It reads a indetifier of node from
   * input stream.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   */
  public void sendTaskID(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    int task_id = -1;

    long node_id = input.readLong();

    if (node_manager.checkNodeID(node_id)) {

      task_id = task_manager.assignTask();

      node_manager.assignTask(node_id, task_id);

      response.setContentType("application/octet-stream");
      response.setHeader("content-disposition",
          "attachment; filename=\"InstanceID\"");
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());
      output.writeInt(task_id);
      output.close();
    } else {
      response.sendError(HttpServletResponse.SC_FORBIDDEN);
    }
  }


  /**
   * Handles <b> stop task </b> request. It reads task identifier from input
   * stream. Stops execution of all nodes working on given task. At the end it
   * writes <code> true </code> in response stream.
   * 
   * @param input
   * @param request
   * @param response
   * @throws IOException
   */
  public void stopTask(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {
    int task_id = input.readInt();
    node_manager.deleteNodesForTask(task_id);
    database_servlets.changeTaskState(task_id, 2);
    task_manager.changeTaskState(task_id, 2);

    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"SendTaskIds\"");

    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());
    output.writeBoolean(true);
    output.close();
  }


  public void sendTaskIds(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {

    List<EvTaskInfo> list = task_manager.getList();

    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"SendTaskIds\"");
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());
    output.writeInt(list.size());

    for (EvTaskInfo info : list) {
      output.writeInt(info.getId());
    }

    output.close();
  }


  public void sendTaskInfo(ObjectInputStream input, HttpServletRequest request,
      HttpServletResponse response) throws IOException {

    int task_id = input.readInt();
    boolean with_jar = input.readBoolean();

    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"sendTaskInfo\"");

    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());

    EvTaskInfo info = task_manager.getTask(task_id);

    output.writeBoolean(info != null);

    output.writeObject(info.getDescription());
    output.writeObject(info.getId());
    output.writeObject(info.getStatus());
    output.writeObject(info.getSubmissionTime());
    if (with_jar) {
      output.writeObject(info.getJar());
    } else {
      output.writeObject(null);
    }

    output.close();
  }


  public void sendNodeCount(ObjectInputStream input,
      HttpServletRequest request, HttpServletResponse response)
      throws IOException {
    int task_id = input.readInt();
    response.setContentType("application/octet-stream");
    response.setHeader("content-disposition",
        "attachment; filename=\"sendNdeCount\"");
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());

    output.writeInt(node_manager.getNodesCountForTask(task_id));

    output.close();
  }

}
