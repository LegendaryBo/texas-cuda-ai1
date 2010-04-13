package pl.wroc.uni.ii.evolution.distribution.strategies.exchange;

import java.util.List;

import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopologyAssigner;
import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Interface of all classes that are used for receive individuals from some
 * source.
 * 
 * @author Marcin Golebiowski, Kamil Dworakowski
 */

public interface EvReceiver<T extends EvIndividual> {

  /**
   * Return best individuals created by neighbours cell
   * 
   * @return List<T>
   */
  List<T> getIndividuals();


  /**
   * Start execution of importer
   * 
   * @return Thread that execute importer
   */
  Thread start();


  /**
   * Inits importer
   * 
   * @param inter
   */
  void init(EvEvolutionInterface inter);


  EvTopologyAssigner getTopology();
}
