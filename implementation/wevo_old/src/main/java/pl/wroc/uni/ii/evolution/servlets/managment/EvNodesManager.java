package pl.wroc.uni.ii.evolution.servlets.managment;

import java.util.Random;

/**
 * @author Marcin Golebiowski
 */
public class EvNodesManager {
  private static final int MAX_CONCURRENT_CONNECTIONS = 1024;

  private static final int TIMEOUT = 60000;

  private long registeredID[] = new long[MAX_CONCURRENT_CONNECTIONS];

  private int task_for_node[] = new int[MAX_CONCURRENT_CONNECTIONS];

  private long lastKeepAlive[] = new long[MAX_CONCURRENT_CONNECTIONS];

  private Random random = new Random(System.currentTimeMillis());


  public EvNodesManager() {
    for (int i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++) {
      registeredID[i] = -1;
      task_for_node[i] = -1;
    }
  }


  /**
   * @param nodeID
   * @return <code> true </code> if node with given <code> nodeID </code> is in
   *         servlet list
   */
  public boolean checkNodeID(long nodeID) {
    for (int i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++)
      if (registeredID[i] == nodeID) {
        lastKeepAlive[i] = System.currentTimeMillis();
        return true;
      }
    return false;
  }


  /**
   * Check timeouts for nodes in servlet list
   */
  public void checkTimeouts() {
    long t = System.currentTimeMillis();

    for (int i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++)
      if (t - lastKeepAlive[i] > TIMEOUT) {
        registeredID[i] = -1;
        task_for_node[i] = -1;
      }
  }


  /**
   * Assigns and returns new node id
   * 
   * @return long
   */
  public long assignNodeID() {
    int i;
    int nodePosition = -1;

    checkTimeouts();

    for (i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++)
      if (registeredID[i] == -1) {
        nodePosition = i;
        break;
      }

    long nodeID = -1;
    if (nodePosition != -1) {
      while (nodeID == -1) {
        nodeID = random.nextLong();
        for (i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++)
          if (registeredID[i] == nodeID)
            nodeID = -1;
      }
      registeredID[nodePosition] = nodeID;
      lastKeepAlive[nodePosition] = System.currentTimeMillis();
    }
    return nodeID;
  }


  /**
   * Store information that node with given <code> node_id </code> is working on
   * task with <code> task_id </code>
   * 
   * @param node_id
   * @param task_id
   */
  public void assignTask(long node_id, int task_id) {
    for (int i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++) {
      if (registeredID[i] == node_id) {
        task_for_node[i] = task_id;
      }
    }
  }


  public void deleteNodesForTask(Integer id) {
    for (int i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++) {
      if (task_for_node[i] == id) {
        registeredID[i] = -1;
      }
    }

  }


  public int getTaskForNode(long node_id) {
    for (int i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++) {
      if (registeredID[i] == node_id) {
        return task_for_node[i];
      }
    }
    return -1;
  }


  public int getNodesCountForTask(int task_id) {
    int count = 0;

    checkTimeouts();

    for (int i = 0; i < MAX_CONCURRENT_CONNECTIONS; i++) {
      if (task_for_node[i] == task_id) {
        count++;
      }
    }
    return count;
  }

}
