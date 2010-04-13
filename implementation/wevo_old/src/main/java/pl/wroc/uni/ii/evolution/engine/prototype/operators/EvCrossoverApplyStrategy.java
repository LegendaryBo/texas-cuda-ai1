package pl.wroc.uni.ii.evolution.engine.prototype.operators;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Objects implementing this interface are used by <b>crossover</b> operators
 * to specify how they going replace old population with new crossed child.<br>
 * For example you may write a strategy to replace all parents, or half of them,
 * or select this randomly.<br>
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public interface EvCrossoverApplyStrategy {

  /**
   * Method called by crossover operators right after apply() call.<br>
   * It's should prepare strategy to work.
   * 
   * @param crossover_arity - crossover operator's arity (how many parents are
   *        needed to perform crossover action)
   * @param combine_result_size - number of children per one crossover action
   * @param population_size - size of input population
   */
  public void init(int crossover_arity, int combine_result_size,
      int population_size);


  /**
   * Specify how many crossover action should be performed during current
   * crossover operator apply.
   * 
   * @return number of crossover action to be performed
   */
  public int getApplyCount();


  /**
   * Defines the way in which children and parent are chosen to be return by the
   * crossover operator.
   * 
   * @param <A>
   * @param child_list - list containing child created in crossover process.
   * @param unselected_parents - list containing parents who weren't used in
   *        crossover process.
   * @return population returned by crossover operator
   */
  public <A extends EvIndividual> List<A> addUnselected(List<A> child_list,
      List<A> unselected);

}
