package pl.wroc.uni.ii.evolution.distribution.strategies.exchange;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopologyAssigner;
import pl.wroc.uni.ii.evolution.distribution.workers.EvBlankEvolInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Default implementation of Importer. Imported individuals are stored in a
 * priority queue. Every <code> delay </code> milliseconds it connects to
 * database through given gateway and fetchs <code> count </code> individuals,
 * then it puts them to queue. If size of queue is greater than
 * <code> size </code> then the <code> (current_size - size) </code> worst
 * individiuals are removed from queue.
 * 
 * @author Kamil Dworakowski, Marcin Golebiowski
 */
public class EvReceiverImpl<T extends EvIndividual> implements EvReceiver<T>,
    Runnable {

  private EvTopologyAssigner topology;

  private EvIndividualsExchanger<T> gateway;

  private EvExchangeQueue<T> queue;

  private int max_queue_size;

  private int delay;

  private long task_id;

  private Thread worker;

  private int count;

  private boolean allows_duplicates;

  private EvEvolutionInterface inter = new EvBlankEvolInterface();


  /**
   * @param gateway a proxy to database
   * @param topology describe how cells are connected
   * @param max_queue_size how many individuals to store in queue
   * @param delay how long to wait between importing individuals from gateway
   *        (ms)
   * @param task_id
   * @param count how many individuals are fetched at most during a single call
   *        to gateway
   * @param allows_duplices specify if duplicate individuals are removed from
   *        queue
   */
  public EvReceiverImpl(EvIndividualsExchanger<T> gateway,
      EvTopologyAssigner topology, int max_queue_size, int delay, long task_id,
      int count, boolean allows_duplicates) {

    this.gateway = gateway;
    this.topology = topology;
    this.max_queue_size = max_queue_size;
    this.delay = delay;
    this.task_id = task_id;
    this.count = count;
    this.allows_duplicates = allows_duplicates;
  }


  /**
   * Inits imporeter
   */
  public void init(EvEvolutionInterface inter) {
    this.inter = inter;
    queue = new EvExchangeQueue<T>(allows_duplicates, max_queue_size);
  }


  /**
   * Removes all individuals from the queue and returns them.
   */
  public List<T> getIndividuals() {
    List<T> collection;
    synchronized (this) {
      collection = new LinkedList<T>(queue);
      queue.clear();
    }
    return collection;
  }


  /**
   * Updates the queue by connecting to database.
   * 
   * @throws IOException
   */
  public void importIndividuals() throws IOException {

    // fetch individuals from gateway
    List<T> imported_individuals = null;

    imported_individuals =
        gateway.importIndividuals(topology.getNeighbours(), task_id, count);
    inter.addImportedPopulation(imported_individuals);

    // updates queue
    synchronized (this) {
      // add imported individuals to queue
      for (T imported_individual : imported_individuals) {
        queue.offer(imported_individual);
      }
    }
  }


  /**
   * Run importer
   */
  public void run() {
    for (;;) {
      try {
        importIndividuals();
        Thread.sleep(delay);
      } catch (InterruptedException e) {
        System.out.println("Importer stopped");
        return;
      } catch (IOException ex) {
        System.out.println("Importer stopped due to problem with connection");
        return;
      }
    }
  }


  /**
   * Start to fetch individuals from server
   */
  public Thread start() {
    worker = new Thread(this);
    worker.start();
    return worker;
  }


  public EvTopologyAssigner getTopology() {
    return topology;
  }

}
