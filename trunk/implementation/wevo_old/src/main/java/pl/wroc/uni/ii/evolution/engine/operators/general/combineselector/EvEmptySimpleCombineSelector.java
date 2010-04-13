// off LineLength
package pl.wroc.uni.ii.evolution.engine.operators.general.combineselector;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCombineParentSelector;

// on LineLength

/**
 * Blank EvCombineParentSelector. It tells crossover not to produce any
 * children. As a result the crossover returns population of parents without any
 * changes.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the operator operates on
 */
class EvEmptySimpleCombineSelector<T extends EvIndividual> implements
    EvCombineParentSelector<T> {

  /** Population algorithm works on. */
  private List<T> population;


  /**
   * {@inheritDoc}
   */
  public List<T> getNextParents() {
    return new ArrayList<T>();
  }


  /**
   * {@inheritDoc}
   */
  public List<T> getUnselectedIndividuals() {
    return population;
  }


  /**
   * {@inheritDoc}
   */
  public void init(final List<T> p, final int parent_count, final int count) {
    this.population = p;
  }
}
