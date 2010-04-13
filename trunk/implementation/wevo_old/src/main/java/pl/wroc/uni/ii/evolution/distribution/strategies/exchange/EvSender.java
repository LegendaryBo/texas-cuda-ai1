package pl.wroc.uni.ii.evolution.distribution.strategies.exchange;

import java.util.List;

import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopologyAssigner;
import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Interface of all classes that are used for send individuals to some
 * destination.
 * 
 * @author Marcin Golebiowski, Kamil Dworakowski
 */

public interface EvSender<T extends EvIndividual> {
  /**
   * Send <code> individuals </code> to some place
   * 
   * @param individuals
   */
  void export(List<T> individuals);


  /**
   * Start execution of exporter
   * 
   * @return Thread that execute exporter
   */
  Thread start();


  /**
   * Inits exporter
   * 
   * @param inter
   */
  void init(EvEvolutionInterface inter);


  long getTaskID();


  long getNodeID();


  EvTopologyAssigner getTopology();

}
