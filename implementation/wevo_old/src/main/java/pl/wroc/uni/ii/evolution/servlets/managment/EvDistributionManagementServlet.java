package pl.wroc.uni.ii.evolution.servlets.managment;

import java.io.*;

import javax.servlet.*;
import javax.servlet.http.*;

/**
 * Basic servlet for managment of distributed evolutionary computation
 * 
 * @author Marcin Golebiowski, Piotr Lipinski
 */
public class EvDistributionManagementServlet extends HttpServlet {

  /**
   * 
   */
  private static final long serialVersionUID = 1L;

  public static final int CMD_GET_NODE_ID = 101;

  private static final int CMD_GET_TASK_ID = 102;

  private static final int CMD_GET_JAR_FILE = 103;

  private static final int CMD_KEEP_ALIVE = 104;

  private static final int CMD_STOP_TASK = 105;

  private static final int CMD_RESUME_TASK = 106;

  private static final int CMD_CLEAR_TASK = 107;

  private static final int CMD_ADD_TASK = 108;

  private static final int CMD_DELETE_TASK = 109;

  private static final int CMD_GET_TASK_IDS = 110;

  private static final int CMD_GET_TASKINFO = 111;

  private static final int CMD_GET_NODES_COUNT = 112;

  private EvDistributionController controller;


  @Override
  public void init(ServletConfig config) throws ServletException {
    super.init(config);
    String wevo_url = config.getInitParameter("WEVO_URL");
    this.controller = new EvDistributionController(wevo_url);
  }


  @Override
  public void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {

    ObjectInputStream input = new ObjectInputStream(request.getInputStream());
    int command = input.readInt();

    switch (command) {
      case CMD_GET_NODE_ID:
        controller.sendNodeID(input, request, response);
        break;
      case CMD_GET_TASK_ID:
        controller.sendTaskID(input, request, response);
        break;
      case CMD_GET_JAR_FILE:
        controller.sendJARFile(input, request, response);
        break;
      case CMD_KEEP_ALIVE:
        controller.sendKeepAlive(input, request, response);
        break;
      case CMD_STOP_TASK:
        controller.stopTask(input, request, response);
        break;
      case CMD_RESUME_TASK:
        controller.resumeTask(input, request, response);
        break;
      case CMD_CLEAR_TASK:
        controller.clearTask(input, request, response);
        break;
      case CMD_DELETE_TASK:
        controller.deleteTask(input, request, response);
        break;

      case CMD_ADD_TASK:
        controller.addTask(input, request, response);
        break;

      case CMD_GET_TASK_IDS:
        controller.sendTaskIds(input, request, response);
        break;

      case CMD_GET_TASKINFO:
        controller.sendTaskInfo(input, request, response);
        break;

      case CMD_GET_NODES_COUNT:
        controller.sendNodeCount(input, request, response);
        break;
    }

    input.close();

  }
}