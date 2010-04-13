package pl.wroc.uni.ii.evolution.engine.operators.general.combineselector;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCombineParentSelector;

/**
 * Simple EvCombineParentSelector for crossover operators working in <br>
 * following way:<br>
 * It selects parents in groups of <b>parent_count</b> starting from the
 * beginning of the input population. Each parent can be selected only once.<br>
 * If last individual in population was selected it returns empty list.<br>
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the operator works on
 */
class EvNonOverlappingSimpleCombineSelector<T extends EvIndividual> implements
    EvCombineParentSelector<T> {

  private List<T> population = null;

  private int parent_count;

  private int next_individual;


  /**
   * {@inheritDoc}
   */
  // method simply return groups of individuals in order from
  // zero index to last index. If there is no more individuals
  // returns empty list
  public List<T> getNextParents() {
    List<T> result = new ArrayList<T>();

    if (next_individual + parent_count <= population.size()) {

      for (int i = next_individual; (i < next_individual + parent_count); i++) {
        result.add(population.get(i));
      }

      next_individual += parent_count;
    }

    return result;
  }


  /**
   * {@inheritDoc}
   */
  // return unselected individuals (which mean individuals last in the list)
  // are returned
  public List<T> getUnselectedIndividuals() {
    List<T> result = new ArrayList<T>();

    for (int j = (population.size() / parent_count) * parent_count; j < population
        .size(); j++) {
      result.add(population.get(j));
    }

    return result;
  }


  /**
   * {@inheritDoc}
   */
  public void init(List<T> population, int parent_count, int count) {
    this.population = population;
    this.parent_count = parent_count;
    this.next_individual = 0;
  }
}
