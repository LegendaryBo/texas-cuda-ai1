package pl.wroc.uni.ii.evolution.engine.operators.general.likeness;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * 
 * Interface containing single method that select the most
 * similar individual from given group to the one specified.<br>
 * It is used in EvRestrictedTournamentReplacement.
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 * @param <T> - type of individual the method match.
 */
public interface EvLikenes<T extends EvIndividual> {

  /**
   * Returns one of individual from given population that is the most
   * similar to the pattern individual.
   * 
   * @param candidates - group from which individuals are matched
   * @param pattern - pattern individual.
   * @return - individual that is the most similar of candidates 
   * population to the patter individual.
   */
  T getSimilar(EvPopulation<T> candidates, T pattern);
  
}
