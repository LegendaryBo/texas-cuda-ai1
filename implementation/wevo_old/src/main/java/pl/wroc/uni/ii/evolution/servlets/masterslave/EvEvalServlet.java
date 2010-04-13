package pl.wroc.uni.ii.evolution.servlets.masterslave;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;

/**
 * Implementation of an servlet for master-slave distribution. It registers
 * nodes, task and assigns jobs for nodes.
 * 
 * @author Kacper Gorski
 */
public class EvEvalServlet extends HttpServlet {

  private static final long serialVersionUID = -3763327879367640508L;

  // comp_node_id -> node_benchmark
  private static HashMap<Long, Double> benchmark_node_map =
      new HashMap<Long, Double>();

  // stores comp_node_id -> last_node_keepalive
  private static HashMap<Long, Long> comp_node_keepalive =
      new HashMap<Long, Long>();

  // stores computration nodes for task
  private static HashMap<Integer, List<Long>> nodes_for_tasks =
      new HashMap<Integer, List<Long>>();

  // stores tasks work_id -> array of individuals ids
  private static HashMap<Long, int[]> current_works =
      new HashMap<Long, int[]>();

  // stores tasks work_id -> array of individuals progress
  private static HashMap<Long, int[]> works_progress =
      new HashMap<Long, int[]>();

  // stores task_id -> identifiers of work_id
  private static HashMap<Integer, PriorityQueue<Long>> seq =
      new HashMap<Integer, PriorityQueue<Long>>();

  private static Long next_work_id = 1L;

  private static Integer synch = 666;

  private static final long timeout = 30 * 1000;

  private static long last_timeout_check = 0;


  /**
   * @param database
   */
  public EvEvalServlet() {
  }


  public void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {

    long curr = System.currentTimeMillis();
    if (last_timeout_check + timeout < curr) {
      checkTimeouts();
      last_timeout_check = curr;
    }

    ObjectInputStream input = new ObjectInputStream(request.getInputStream());
    ObjectOutputStream output =
        new ObjectOutputStream(response.getOutputStream());
    int command = input.readInt();

    switch (command) {
      case EvEvalServletOperations.CMD_ADD_WORK:
        addWork(input, output);
        break;
      case EvEvalServletOperations.CMD_REGISTER:
        register(input, output);
        break;
      case EvEvalServletOperations.CMD_UNREGISTER:
        unregister(input, output);
        break;
      case EvEvalServletOperations.CMD_GET_UNEVALED_WORK:
        getUnevaledTask(input, output);
        break;
      case EvEvalServletOperations.CMD_INFORM:
        inform(input, output);
        break;
      case EvEvalServletOperations.CMD_IS_WORK_END:
        isEnd(input, output);
        break;
      case EvEvalServletOperations.CMD_DELETE:
        deleteWork(input, output);
        break;
      case EvEvalServletOperations.CMD_COMP_NODE_KEEP_ALIVE:
        keepalive(input, output);
        break;
      case EvEvalServletOperations.CMD_GET_WORKS:
        getWorks(input, output);
        break;

      case EvEvalServletOperations.CMD_GET_NODES_FOR_TASK:
        getNodesForTask(input, output);
    }
    input.close();
    output.close();
  }


  private void getNodesForTask(ObjectInputStream input,
      ObjectOutputStream output) throws IOException {

    int task_id = input.readInt();

    Long[] tmp;

    synchronized (synch) {
      tmp =
          nodes_for_tasks.get(task_id).toArray(
              new Long[nodes_for_tasks.get(task_id).size()]);
    }

    output.writeInt(tmp.length);

    for (int i = 0; i < tmp.length; i++) {
      output.writeLong(tmp[i]);
    }
  }


  private void getWorks(ObjectInputStream input, ObjectOutputStream output)
      throws IOException {
    Set<Long> keys = null;

    synchronized (synch) {
      keys = current_works.keySet();
    }

    output.writeInt(keys.size());
    for (Long work_id : keys) {
      output.writeLong(work_id);
    }
  }


  private void checkTimeouts() {

    HashMap<Integer, List<Long>> tmp = new HashMap<Integer, List<Long>>();

    synchronized (synch) {
      for (Integer task : nodes_for_tasks.keySet()) {
        List<Long> new_nodes = new ArrayList<Long>();
        for (Long node_id : nodes_for_tasks.get(task)) {
          if (comp_node_keepalive.get(node_id) + timeout >= System
              .currentTimeMillis()) {
            new_nodes.add(node_id);
          } else {
            try {
              comp_node_keepalive.remove(node_id);
              benchmark_node_map.remove(node_id);
            } catch (Exception ex) {
              ex.printStackTrace(System.out);
            }
          }
        }
        tmp.put(task, new_nodes);
      }
      nodes_for_tasks = tmp;
    }
  }


  private void keepalive(ObjectInputStream input, ObjectOutputStream output)
      throws IOException {
    long comp_node_id = input.readLong();
    boolean res = true;
    synchronized (synch) {
      comp_node_keepalive.put(comp_node_id, System.currentTimeMillis());
    }
    output.writeBoolean(res);
  }


  private void deleteWork(ObjectInputStream input, ObjectOutputStream output)
      throws IOException {

    int task_id = input.readInt();
    long work_id = input.readLong();

    boolean result = true;

    synchronized (synch) {
      try {
        seq.get(task_id).remove(work_id);
        current_works.remove(work_id);
        works_progress.remove(work_id);
      } catch (Exception ex) {
        result = false;

      }
    }
    output.writeBoolean(result);
  }


  private void isEnd(ObjectInputStream input, ObjectOutputStream output)
      throws IOException {
    long work_id = input.readLong();
    boolean result = true;

    synchronized (synch) {
      int[] progress = works_progress.get(work_id);

      for (int i = 0; i < progress.length; i++) {
        if (progress[i] != 2) {
          result = false;
          break;
        }
      }
    }

    output.writeBoolean(result);
  }


  private void register(ObjectInputStream input, ObjectOutputStream output)
      throws IOException {

    int task_id = input.readInt();
    long node_id = input.readLong();
    double benchmark = input.readDouble();
    boolean result = true;

    synchronized (synch) {
      try {
        if (benchmark_node_map.containsKey(node_id)) {
          throw new Exception();
        }
        // add to benchmark node map
        benchmark_node_map.put(node_id, benchmark);

        // add node to nodes for task
        if (nodes_for_tasks.containsKey(task_id)) {

          if (nodes_for_tasks.get(task_id).contains(node_id)) {
            throw new Exception();
          }
          nodes_for_tasks.get(task_id).add(node_id);
        } else {
          List<Long> res = new ArrayList<Long>();
          res.add(node_id);
          nodes_for_tasks.put(task_id, res);
        }

        // add to keepalive map

        if (comp_node_keepalive.containsKey(node_id)) {
          throw new Exception();
        }
        comp_node_keepalive.put(node_id, System.currentTimeMillis());

      } catch (Exception ex) {
        result = false;
        ex.printStackTrace(System.out);
      }
    }

    // write result
    output.writeBoolean(result);
  }


  //
  private void unregister(ObjectInputStream input, ObjectOutputStream output)
      throws IOException {
    int task_id = input.readInt();
    long node_id = input.readLong();
    boolean result = true;

    synchronized (synch) {

      try {
        if (!benchmark_node_map.containsKey(node_id)) {
          result = false;
        }

        // remove from benchmar_node_map
        benchmark_node_map.remove(node_id);

        if (!comp_node_keepalive.containsKey(node_id)) {
          result = false;
        }

        // remove from keepalive_map
        comp_node_keepalive.remove(node_id);

        if (!nodes_for_tasks.get(task_id).contains(node_id)) {
          result = false;
        }

        // remove from node for task
        nodes_for_tasks.get(task_id).remove(node_id);

      } catch (Exception ex) {
        result = false;
      }
    }
    output.writeBoolean(result);
  }


  private void addWork(ObjectInputStream input, ObjectOutputStream output)
      throws IOException {

    boolean result = true;
    int task_id = input.readInt(); // first get task_id
    int work_size = input.readInt(); // secodnly size of the table

    // get indentifiers of individuals
    int[] work = new int[work_size];
    for (int i = 0; i < work_size; i++) {
      work[i] = input.readInt();
    }

    synchronized (synch) {
      try {

        // add to current_tasks
        current_works.put(next_work_id, work);
        works_progress.put(next_work_id, new int[work_size]);

        // add to task sequence
        if (seq.containsKey(task_id)) {
          seq.get(task_id).offer(next_work_id);
        } else {
          PriorityQueue<Long> queue = new PriorityQueue<Long>();
          queue.add(next_work_id);
          seq.put(task_id, queue);
        }

      } catch (Exception ex) {
        result = false;
      }

      if (result) {
        output.writeLong(next_work_id);
      } else {
        output.writeLong(0);
      }
      next_work_id++;
    }
  }


  // returns node_id, and list of identifiers
  private void getUnevaledTask(ObjectInputStream input,
      ObjectOutputStream output) throws IOException {

    // read task_id
    int task_id = input.readInt();

    // read computation node identifier
    long comp_node_id = input.readLong();

    // variables
    List<Integer> given_work_indexes = new ArrayList<Integer>();
    double node_benchmark = 0;
    double total_benchmark = 0;
    int[] work = null;
    int[] work_progress = null;
    Long work_id = 0L;
    boolean work_to_do = true;

    synchronized (synch) {

      try {
        // there is no work for task
        if (!seq.containsKey(task_id) || !nodes_for_tasks.containsKey(task_id)
            || seq.get(task_id).size() == 0) {
          throw new Exception();
        } else {
          // get node_benchmark
          node_benchmark = benchmark_node_map.get(comp_node_id);

          // compute current total benchmark
          total_benchmark = 0;
          for (Long node_id : nodes_for_tasks.get(task_id)) {
            total_benchmark += benchmark_node_map.get(node_id);
          }

          // get current work for task
          work_id = seq.get(task_id).peek();
          work = current_works.get(work_id);
          work_progress = works_progress.get(work_id);
        }

        int work_size = work.length;
        // number of individuals that this node will receive to evaluate
        int max_to_be_evaluated =
            1 + (int) (work_size * (node_benchmark / total_benchmark));

        // firstly we look for individuals that havent been sent yet
        for (int i = 0; (i < work_size); i++) {
          if (work_progress[i] == 0) {
            work_progress[i] = 1;
            given_work_indexes.add(work[i]);
            if (given_work_indexes.size() >= max_to_be_evaluated) {
              break;
            }
          }
        }

        // if there are no individuals that havent been sent yet, retrieve those
        // that
        // are undone, but in progress in another node
        if (given_work_indexes.size() == 0) {

          for (int i = 0; i < work_size; i++) {
            if (work_progress[i] == 1) {
              given_work_indexes.add(work[i]);
              if (given_work_indexes.size() >= max_to_be_evaluated)
                break;
            }
          }
        }

        // if no work given (everything is done), return
        if (given_work_indexes.size() == 0) {
          seq.get(task_id).remove(work_id);
        }

      } catch (Exception ex) {
        // ex.printStackTrace(System.out);
        work_to_do = false;
      }
    }

    // returning result to the client
    if (work_to_do) {
      // if there is work
      output.writeLong(work_id);
      output.writeInt(given_work_indexes.size());
      for (int i = 0; i < given_work_indexes.size(); i++) {
        output.writeInt(given_work_indexes.get(i));
      }
    } else {
      // System.out.println("Brak zadañ");
      // if there isn't work
      output.writeLong(0);
    }
  }


  /**
   * Sent an informations about indexes of individuals that have been evaluated.<BR>
   */
  private void inform(ObjectInputStream input, ObjectOutputStream output)
      throws IOException {

    boolean result = true;

    long node_id = input.readLong();

    int work_size = input.readInt();
    int[] work_done = new int[work_size];

    for (int i = 0; i < work_size; i++) {
      work_done[i] = input.readInt();
    }

    TreeSet<Integer> set = new TreeSet<Integer>();
    for (int i = 0; i < work_done.length; i++) {
      set.add(work_done[i]);
    }

    synchronized (synch) {

      try {
        int[] work = current_works.get(node_id);
        int[] progress = works_progress.get(node_id);

        for (int i = 0; i < work.length; i++) {
          if (set.contains(work[i])) {
            progress[i] = 2;
          }
        }
        works_progress.put(node_id, progress);
      } catch (Exception ex) {
        result = false;
      }
    }
    output.writeBoolean(result);
  }
}