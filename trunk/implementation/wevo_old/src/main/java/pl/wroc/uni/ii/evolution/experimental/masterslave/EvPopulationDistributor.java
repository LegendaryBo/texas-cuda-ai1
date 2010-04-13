/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package pl.wroc.uni.ii.evolution.experimental.masterslave;

import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Strategy that specifies how to distribute population 
 * between slaves.
 * 
 * @author Karol 'Asgaroth' Stosiek (karol.stosiek@gmail.com)
 * @author Mateusz 'm4linka' Malinowski (m4linka@gmail.com)
 * @param <T> - subtype of EvIndividual
 */
public interface EvPopulationDistributor <T extends EvIndividual> {

  /**
   * Distributes population among clients.
   * 
   * @param population - population to distribute
   * @param clients - client's id list
   */
  void distribute(final EvPopulation<T> population, 
      final List<EvClientID> clients);
  
  /**
   * Returns subpopulation which is assigned to the client id.
   * 
   * @param client_id - client id
   * @return subpopulation which is assigned to the client_id
   */
  EvPopulation<T> getSubpopulation(final EvClientID client_id);
  
  /**
   * Adds (evaluated?) subpopulation to the result.
   * 
   * @param subpopulation - subpopulation to set
   * @param client_id - client id
   */
  void setSubpopulation(
      final EvPopulation<T> subpopulation, final EvClientID client_id);
  
  /**
   * Returns the whole population.
   * 
   * @return the whole population
   */
  EvPopulation<T> getPopulation();
  
}
