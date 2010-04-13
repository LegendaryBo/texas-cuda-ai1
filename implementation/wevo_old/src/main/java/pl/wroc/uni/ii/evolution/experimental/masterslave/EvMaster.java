package pl.wroc.uni.ii.evolution.experimental.masterslave;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Master class in master-slave model.
 * 
 * @author Karol 'Asgaroth' Stosiek (karol.stosiek@gmail.com)
 * @author Mateusz 'm4linka' Malinowski (m4linka@gmail.com)
 * @param <T> - subtype of EvIndividual
 */
public class EvMaster<T extends EvIndividual> {
  
  /**
   * Evaluator which evaluates populations.
   */
  private EvEvaluator<T> evaluator;
  
  /**
   * Algorithm which is used in evolution.
   */
  private EvAlgorithm algorithm;
  
  /**
   * Database connector.
   */
  private EvDBConnector db_connector;
  
  /**
   * Constructor.
   * 
   * @param ev_algorithm - evolutionary algorithm to set
   * @param ev_population_distributor - distributor to set
   */
  public EvMaster(
      final EvAlgorithm ev_algorithm,
      final EvPopulationDistributor<T> ev_population_distributor) {
    this.algorithm = ev_algorithm;
    this.evaluator = new EvEvaluator(ev_population_distributor);
    this.db_connector = null;
  }
  
  /**
   * Sets db connector.
   * 
   * @param ev_db_connector - database connector to set
   */
  public void setDBConnector(final EvDBConnector ev_db_connector) {
    // TODO: implement it
  }
  
  /**
   * Main method, do distributed evolution.
   */
  public void run() {
    while (!this.algorithm.isTerminationConditionSatisfied()) {
      this.algorithm.doIteration();
      EvPopulation population = 
          this.evaluator.evaluate(this.algorithm.getPopulation());
      this.algorithm.setPopulation(population);
      
      // do something with database and population
     }
  }
}
