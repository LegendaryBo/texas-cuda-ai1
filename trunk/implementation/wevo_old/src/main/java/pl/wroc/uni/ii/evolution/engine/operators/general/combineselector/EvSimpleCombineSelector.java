package pl.wroc.uni.ii.evolution.engine.operators.general.combineselector;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCombineParentSelector;

/**
 * Simple implementation of EvCombineParentSelector. <br>
 * It uses one of 3 EvCombineParentSelector depending on population size and
 * number of combine() calls:<br>
 * 1) If there is not enough parent in input population to perform single
 * crossover action - use EvEmptySimpleCombineSelector (which does nothing)<br>
 * 2) If there is need to call at least one parent more than once
 * (population_size * number_of_crossover > output_population) - use
 * EvOverlappingSimpleCombineSelector<br>
 * 3) If there is no need to use all individuals in population as parents
 * (population_size * number_of_crossover <= output_population) - use
 * EvNonOverlappingSimpleCombineSelector<br>
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the operator works on
 */
public class EvSimpleCombineSelector<T extends EvIndividual> implements
    EvCombineParentSelector<T> {

  private EvCombineParentSelector<T> appropriate = null;


  /**
   * Returns next individuals to combine
   */
  public List<T> getNextParents() {
    return appropriate.getNextParents();
  }


  /**
   * Returns unselected individuals as combine argument
   */
  public List<T> getUnselectedIndividuals() {
    return appropriate.getUnselectedIndividuals();
  }


  /**
   * Inits EvSimpleCombineSelector. Using given parameters customizes its
   * behavior.
   */
  public void init(List<T> population, int parent_count, int count) {
    // when there is not enough parent, assign blank selector
    if (parent_count > population.size()) {
      appropriate = new EvEmptySimpleCombineSelector<T>();
    } else {
      // when there is need to select at least one parent more than once
      // to fill all population with children
      if (parent_count * count <= population.size()) {

        appropriate = new EvNonOverlappingSimpleCombineSelector<T>();

        // when it's only need to select only some of parents of the population
      } else {
        appropriate = new EvOverlappingSimpleCombineSelector<T>();
      }
    }

    appropriate.init(population, parent_count, count);

  }

}
