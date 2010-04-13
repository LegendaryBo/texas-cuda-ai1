package pl.wroc.uni.ii.evolution.distribution.strategies;

import java.util.List;

import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchangeWithServlet;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchanger;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiver;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiverImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvSender;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.ExSenderImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopology;
import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationWithErrorRecovery;

/**
 * Object explaining island model used in distribution.<br>
 * This object explains the way individuals are exchanged between nodes in
 * island model.<br>
 * Following rules can be modified:<br> - Merger, EvReplacement object which
 * decide how incoming individuals will be merged with this node's population<br> -
 * Export selector, EvOperator object that decide which individuals shall be
 * sent to neighbor nodes.<br>
 * <br>
 * If merger and export selector are not set, EvBestFromUnionReplacement and new
 * EvKBestSelection(10) operators are set by default.<br>
 * It doesn't not contain information about distributed algorithm.<br>
 * <br>
 * If you aren't familiar with island model, created object using simplified
 * constructor.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @author Kamil Dworakowski
 * @author Kacper Gorski (admin@34all.pl)
 */
public class EvIslandModel<T extends EvIndividual> {

  private EvReceiver<T> receiver;

  private EvSender<T> sender;

  private EvReplacement<T> merger;

  private EvOperator<T> export_selection;

  private Thread[] threads = new Thread[2];


  /**
   * Constructor for DistributionStrategy
   * 
   * @param receiver fetch individuals from other cells
   * @param sender send individuals to other cells
   * @param merger merge fetched individuals with current population
   * @param export_selection specify what individuals are send to other cells
   * @deprecated WILL SOON BE DELETED
   */
  public EvIslandModel(EvReceiver<T> receiver, EvSender<T> sender,
      EvReplacement<T> merger, EvOperator<T> export_selection) {
    this.receiver = receiver;
    this.sender = sender;
    this.merger = merger;
    this.export_selection = export_selection;
  }


  /**
   * Simplified constructor which uses default replacement and selection. It
   * constructs itself EvBestFromUnionReplacement operator and
   * EvKBestSelection(10) operator.
   * 
   * @param receiver fetch individuals from other cells
   * @param sender send individuals to other cells
   * @deprecated WILL SOON BE DELETED
   */
  public EvIslandModel(EvReceiver<T> receiver, EvSender<T> sender) {
    this(receiver, sender, new EvBestFromUnionReplacement<T>(),
        new EvKBestSelection<T>(10));
  }


  /**
   * Default constructor that creates model island distribution. <br>
   * It can be defined here how many individuals (and which individuals) are
   * send between nodes, the delay between individual exchanges and operators
   * that merges them with node's population
   * 
   * @param wevo_url - server address
   * @param topology - topology of model island distribution
   * @param merger - operator that merges individuals incoming from neighbor
   *        nodes and current node population node.
   * @param export_selector - operator that selects which individual are sent to
   *        other nodes
   * @param task_id -
   * @param export_population_size - how many individuals are exported to other
   *        nodes
   * @param export_delay - delay between different packages of imported
   *        individuals
   * @param import_population_size - how many individuals are imported from
   *        other nodes
   * @param import_delay - delay between different packages of exported
   *        individuals
   * @param export_duplicates - false, equal individual's won't be exported
   * @param import_duplicates - false, equal individual's won't be imported
   */
  public EvIslandModel(String wevo_url, EvTopology topology,
      EvReplacement merger, EvOperator export_selection, long task_id,
      int export_population_size, int export_delay, int import_population_size,
      int import_delay, boolean export_duplicates, boolean import_duplicates) {

    EvIndividualsExchanger<EvPermutationIndividual> exchanger =
        new EvIndividualsExchangeWithServlet<EvPermutationIndividual>(
            new EvDBServletCommunicationWithErrorRecovery(wevo_url, 5, 2000));
    long node_id = topology.assignCellID();

    this.receiver =
        new EvReceiverImpl(exchanger, topology, 20, import_delay, task_id,
            import_population_size, import_duplicates);
    this.sender =
        new ExSenderImpl(exchanger, topology, 20, export_delay, task_id,
            node_id, export_population_size, export_duplicates);
    this.merger = merger;
    this.export_selection = export_selection;

  }


  /**
   * Creates island model distribution with EvBestFromUnionReplacement as merger
   * (operator which merges incoming population with algorithm's population) and
   * EvKBestSelection(10) as export_selector (the object that choose which
   * individuals to send)
   * 
   * @param wevo_url - server address
   * @param topology - topology of model island distribution
   * @param task_id -
   * @param export_population_size - how many individuals are exported to other
   *        nodes
   * @param export_delay - delay between different packages of imported
   *        individuals
   * @param import_population_size - how many individuals are imported from
   *        other nodes
   * @param import_delay - delay between different packages of exported
   *        individuals
   * @param export_duplicates - false, equal individual's won't be exported
   * @param import_duplicates - false, equal individual's won't be imported
   */
  public EvIslandModel(String wevo_url, EvTopology topology, long task_id,
      int export_population_size, int export_delay, int import_population_size,
      int import_delay, boolean export_duplicates, boolean import_duplicates) {

    this(wevo_url, topology, new EvBestFromUnionReplacement<T>(),
        new EvKBestSelection<T>(10), task_id, export_population_size,
        export_delay, import_population_size, import_delay, export_duplicates,
        import_duplicates);
  }


  /**
   * Simplified constructor which creates simple island model distribution. It
   * uses default replacement and selection. It constructs itself
   * EvBestFromUnionReplacement operator and EvKBestSelection(10) operator.<br>
   * Every node send and receive 10 individuals per 10 seconds. <br>
   * These values can be changed by calling complex constructor.
   * 
   * @param wevo_url - server URL.
   * @param topology - topology of island model.
   * @param task_id
   */
  public EvIslandModel(String wevo_url, EvTopology topology, long task_id) {
    this(wevo_url, topology, new EvBestFromUnionReplacement<T>(),
        new EvKBestSelection<T>(10), task_id, 10, 10000, 10, 10000, false,
        false);
  }


  /**
   * It starts distribution strategy.
   */
  public void init(EvEvolutionInterface inter) {
    receiver.init(inter);
    threads[0] = receiver.start();
    sender.init(inter);
    threads[1] = sender.start();
  }


  /**
   * Check if both object's threads are ok.
   * 
   * @return true if everything is ok., false othervise.
   */
  public boolean isOkey() {
    return threads[0].isAlive() && threads[1].isAlive();
  }


  /**
   * Stops the object
   */
  public void stop() {

    System.out.println("Stopping Importer and Exporter");
    threads[0].interrupt();
    threads[1].interrupt();

    try {
      threads[0].join();
      threads[1].join();
      System.out.println("Importer and Exporter stopped");
    } catch (InterruptedException e) {
      System.out
          .println("Error: Joining interupted threads for Importer and Exporter failed");
      return;
    }

  }


  /**
   * The sane as public void updatePopulation(EvPopulation<T> population), but
   * this fucntion sends an population to interface
   * 
   * @param population
   */
  public void updatePopulation(EvPopulation<T> population) {

    // get individual from neighbours
    List<T> from_neighbours = receiver.getIndividuals();

    // create new population
    EvPopulation<T> temp = new EvPopulation<T>(from_neighbours);
    EvPopulation<T> new_population = merger.apply(population, temp);

    // replace old population
    population.clear();
    population.addAll(new_population);

  }


  /**
   * The same as public void export(EvPopulation<T> population,
   * EvEvolutionInterface inter). This function also send population to
   * Interface
   * 
   * @param population
   */
  public void export(EvPopulation<T> population) {
    // System.out.println("Export selection start");
    sender.export(export_selection.apply(population));
    // System.out.println("Export selection end");
  }


  /**
   * @return current EvSender object
   */
  public EvSender getSender() {
    return sender;
  }

}
