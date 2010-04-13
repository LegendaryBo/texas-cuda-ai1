package pl.wroc.uni.ii.evolution.distribution.strategies.exchange;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopologyAssigner;
import pl.wroc.uni.ii.evolution.distribution.workers.EvBlankEvolInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Default implementation for EvSender
 * 
 * @author Kamil Dworakowski, Marcin Golebiowski
 * @param <T>
 */
public class ExSenderImpl<T extends EvIndividual> implements EvSender<T>,
    Runnable {

  private EvIndividualsExchanger<T> gateway;

  private EvTopologyAssigner topology;

  private int max_queue_size;

  private int send_size;

  private long task_id;

  private long node_id;

  private long delay;

  private Thread worker;

  private boolean allows_duplicates;

  private EvExchangeQueue<T> queue;

  private EvEvolutionInterface inter = new EvBlankEvolInterface();


  /**
   * @param gateway a proxy to database
   * @param topology
   * @param max_queue_size how many individuals are stored in send queue
   * @param delay how long to wait between exporting individuals through gateway
   *        (ms)
   * @param task_id
   * @param node_id
   * @param send_size how many individuals are sended at most at single call to
   *        gateway
   * @param allows_duplicates specify if duplicate individuals are removed from
   *        queue
   */
  public ExSenderImpl(EvIndividualsExchanger<T> gateway,
      EvTopologyAssigner topology, int max_queue_size, long delay,
      long task_id, long node_id, int send_size, boolean allows_duplicates) {
    super();
    this.gateway = gateway;
    this.topology = topology;
    this.max_queue_size = max_queue_size;
    this.task_id = task_id;
    this.node_id = node_id;
    this.delay = delay;
    this.allows_duplicates = allows_duplicates;
    this.send_size = send_size;
  }


  /**
   * {@inheritDoc}
   */
  public void init(EvEvolutionInterface inter) {
    this.inter = inter;
    queue = new EvExchangeQueue<T>(this.allows_duplicates, this.max_queue_size);
  }


  /**
   * Add individuals to send buffer
   */
  public void export(List<T> individuals) {
    synchronized (this) {
      queue.addAll(individuals);
    }
  }


  /**
   * Start to send individuals to server
   */
  public Thread start() {
    worker = new Thread(this);
    worker.start();
    return worker;
  }


  public void run() {
    for (;;) {
      try {
        sendSomeIndividuals();
        Thread.sleep(delay);
      } catch (InterruptedException e) {
        System.out.println("Exporter stopped");
        return;
      } catch (IOException ex) {
        System.out.println("Exporter stopped due to problem with connection");
        return;
      }
    }
  }


  /**
   * Sends <code> send_size </code> individuals from queue
   * 
   * @throws IOException
   */
  public void sendSomeIndividuals() throws IOException {

    List<T> individuals_to_send = new ArrayList<T>();
    synchronized (this) {
      int sended = 0;
      while ((sended < send_size) && (!queue.isEmpty())) {
        individuals_to_send.add(queue.poll());
        sended++;
      }
    }
    inter.addExportedPopulation(individuals_to_send);
    gateway.exportIndividuals(task_id, topology.assignCellID(), node_id,
        individuals_to_send);
  }


  public long getNodeID() {
    return node_id;
  }


  public long getTaskID() {
    return task_id;
  }


  public EvTopologyAssigner getTopology() {
    return topology;
  }
}