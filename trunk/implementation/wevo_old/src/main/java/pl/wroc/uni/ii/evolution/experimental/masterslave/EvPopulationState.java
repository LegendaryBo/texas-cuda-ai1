package pl.wroc.uni.ii.evolution.experimental.masterslave;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Wrapper of the population class.
 * 
 * @author Karol 'Asgaroth' Stosiek (karol.stosiek@gmail.com)
 * @author Mateusz 'm4linka' Malinowski (m4linka@gmail.com)
 * @param <T> - subtype of EvIndividual
 */
final class EvPopulationState <T extends EvIndividual> {
  
  /**
   * True if population is evaluated.
   */
  private boolean is_evaluated;
  
  /**
   * Population.
   */
  private  EvPopulation population;

  /**
   * Constructor.
   * 
   * @param ev_population - population
   */
  public EvPopulationState(final EvPopulation<T> ev_population) {
    this.is_evaluated = false;
    this.population = ev_population;
  }

  /**
   * 
   * @return true if population is evaluated, false otherwise
   */
  public boolean isEvaluated() {
    return is_evaluated;
  }

  /**
   * Sets population evaluation.
   * 
   * @param ev_is_evaluated - true if population is evaluated, false otherwise
   */
  public void setEvaluated(final boolean ev_is_evaluated) {
    this.is_evaluated = ev_is_evaluated;
  }

  /**
   * 
   * @return population
   */
  public EvPopulation getPopulation() {
    return population;
  }

  /**
   * 
   * @param ev_population - population to set
   */
  public void setPopulation(final EvPopulation ev_population) {
    this.population = ev_population;
  }
}
