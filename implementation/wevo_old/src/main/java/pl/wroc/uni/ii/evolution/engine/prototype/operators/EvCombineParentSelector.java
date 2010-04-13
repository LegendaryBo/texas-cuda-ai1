package pl.wroc.uni.ii.evolution.engine.prototype.operators;

import java.util.List;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * An interface for all classes used by EvCrossover operator to select
 * individuals to combine.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals EvCombineParentSelector works on
 */
public interface EvCombineParentSelector<T extends EvIndividual> {

  /**
   * Initializes EvParentIterator.
   * 
   * @param population a source of individuals
   * @param parent_count how many parents are needed to crossover operator.
   * @param count - how many times will combine be called
   */
  public void init(List<T> population, int parent_count, int count);


  /**
   * Returns next parents. If there aren't next parenets then empty list is
   * returned.
   * 
   * @return some individuals
   */
  public List<T> getNextParents();


  /**
   * Returns unselected individuals. If every individuals in population are used
   * as parent then empty list is returned.
   * 
   * @return some individuals
   */
  public List<T> getUnselectedIndividuals();

}
