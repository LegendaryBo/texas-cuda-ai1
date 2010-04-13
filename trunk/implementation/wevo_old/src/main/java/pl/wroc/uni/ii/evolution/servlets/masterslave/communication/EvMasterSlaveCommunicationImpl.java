package pl.wroc.uni.ii.evolution.servlets.masterslave.communication;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.distribution.tools.EvDistributionTools;
import pl.wroc.uni.ii.evolution.servlets.masterslave.EvEvalServletOperations;
import pl.wroc.uni.ii.evolution.servlets.masterslave.EvWorkInfo;

public class EvMasterSlaveCommunicationImpl implements
    EvMasterSlaveCommunication {

  private String servlet_url;


  /**
   * @param wevo_server_url
   */
  public EvMasterSlaveCommunicationImpl(String wevo_server_url) {
    this.servlet_url = EvDistributionTools.eval_servlet_url(wevo_server_url);
  }


  public long addWork(int task_id, int[] ids) throws Exception {
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvEvalServletOperations.CMD_ADD_WORK);
    output.writeInt(task_id);
    output.writeInt(ids.length);

    for (int i = 0; i < ids.length; i++) {
      output.writeInt(ids[i]);
    }
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    long work_id = input.readLong();
    input.close();

    return work_id;
  }


  public boolean deleteWork(int task_id, long work_id) throws Exception {

    boolean res = true;
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvEvalServletOperations.CMD_DELETE);
    output.writeInt(task_id);
    output.writeLong(work_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    res = input.readBoolean();
    input.close();

    return res;
  }


  public EvWorkInfo getWork(int task_id, long comp_node_id) throws Exception {

    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvEvalServletOperations.CMD_GET_UNEVALED_WORK);
    output.writeInt(task_id);
    output.writeLong(comp_node_id);
    output.close();
    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());

    long work_id = input.readLong();

    if (work_id == 0) {
      return null;
    }

    int len = input.readInt();
    int[] ids = new int[len];
    for (int i = 0; i < len; i++) {
      ids[i] = input.readInt();
    }
    input.close();

    EvWorkInfo info = new EvWorkInfo(task_id, work_id, ids);

    return info;
  }


  public boolean informWorkDone(EvWorkInfo work) throws Exception {

    boolean res = true;
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvEvalServletOperations.CMD_INFORM);
    output.writeLong(work.work_id);
    output.writeInt(work.ids.length);

    for (int i = 0; i < work.ids.length; i++) {
      output.writeInt(work.ids[i]);
    }

    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    res = input.readBoolean();
    input.close();

    return res;
  }


  public boolean isWorkDone(long work_id) throws Exception {
    boolean res = true;
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvEvalServletOperations.CMD_IS_WORK_END);
    output.writeLong(work_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    res = input.readBoolean();
    input.close();

    return res;
  }


  public boolean registerComputaionNode(int task_id, long comp_node_id,
      double benchmark) throws Exception {

    boolean res = true;
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvEvalServletOperations.CMD_REGISTER);
    output.writeInt(task_id);
    output.writeLong(comp_node_id);
    output.writeDouble(benchmark);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    res = input.readBoolean();
    input.close();

    return res;
  }


  public boolean unregisterComputationNode(int task_id, long comp_node_id)
      throws Exception {
    boolean res = true;
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());

    output.writeInt(EvEvalServletOperations.CMD_UNREGISTER);
    output.writeInt(task_id);
    output.writeLong(comp_node_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    res = input.readBoolean();
    input.close();

    return res;
  }


  public long[] getWorks() throws IOException {
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(EvEvalServletOperations.CMD_GET_WORKS);
    output.close();

    List<Long> work_ids = new ArrayList<Long>();
    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int len = input.readInt();

    for (int i = 0; i < len; i++) {
      work_ids.add(input.readLong());
    }
    input.close();

    long[] res = new long[work_ids.size()];

    for (int i = 0; i < work_ids.size(); i++) {
      res[i] = work_ids.get(i);
    }
    return res;
  }


  public boolean keepalive(long comp_node_id) throws Exception {
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(EvEvalServletOperations.CMD_COMP_NODE_KEEP_ALIVE);
    output.writeLong(comp_node_id);
    output.close();

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());

    boolean res = input.readBoolean();

    input.close();

    return res;

  }


  public long[] getNodesForTask(int task_id) throws Exception {
    URL servlet = new URL(servlet_url);
    URLConnection connection = servlet.openConnection();
    connection.setDoOutput(true);
    ObjectOutputStream output =
        new ObjectOutputStream(connection.getOutputStream());
    output.writeInt(EvEvalServletOperations.CMD_GET_NODES_FOR_TASK);
    output.writeInt(task_id);
    output.close();

    List<Long> nodes_id = new ArrayList<Long>();
    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());
    int len = input.readInt();

    for (int i = 0; i < len; i++) {
      nodes_id.add(input.readLong());
    }
    input.close();

    long[] res = new long[nodes_id.size()];

    for (int i = 0; i < nodes_id.size(); i++) {
      res[i] = nodes_id.get(i);
    }
    return res;
  }
}
