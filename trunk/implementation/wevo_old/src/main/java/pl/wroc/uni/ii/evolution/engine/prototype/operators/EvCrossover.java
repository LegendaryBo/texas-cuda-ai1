package pl.wroc.uni.ii.evolution.engine.prototype.operators;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.combineselector.EvSimpleCombineSelector;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * An abstract class for every crossover operators. <br>
 * Unlike in other operators you don't have to overwrite apply() method.
 * Instead, apply() is already implemented, it produces new individuals by
 * multiply calling combine() method.<br>
 * You can specify: <br> - strategy for applying this operator and use <br> -
 * combine parent selector: <br>
 * <br>
 * An operator extending this class will call combine() as many times as it's
 * defined by EvCrossoverApplyStrategy (by default
 * EvPersistPopulationSizeApplyStrategy, to change it call
 * setCrossoverStrategy() )<br>
 * Parent used to create new individuals are defined by EvCombineParentSelector
 * object which can be set using setCombineParentSelector (default
 * EvSimpleCombineSelector)
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the crossover operates on
 */
public abstract class EvCrossover<T extends EvIndividual> implements
    EvOperator<T> {

  private EvCrossoverApplyStrategy crossover_strategy = null;

  private EvCombineParentSelector<T> combine_parent_selector = null;


  public EvPopulation<T> apply(EvPopulation<T> population) {

    /** if null then set default EvCombineSelector * */
    if (combine_parent_selector == null) {
      combine_parent_selector = new EvSimpleCombineSelector<T>();
    }

    /** if null then set default EvCrossoverStrategy * */
    if (crossover_strategy == null) {
      crossover_strategy = new EvPersistPopulationSizeApplyStrategy();
    }

    /** init crossover_stategy * */
    crossover_strategy.init(arity(), combineResultSize(), population.size());

    /** compute how many in this strategy combine will be applied to parents * */
    int apply_count = crossover_strategy.getApplyCount();

    /** inits combine_parent_selector * */
    combine_parent_selector.init(population, arity(), apply_count);

    List<T> result = new ArrayList<T>();

    /** creating children * */
    for (int i = 0; i < apply_count; i++) {

      List<T> inviduals_to_combine = combine_parent_selector.getNextParents();

      if (inviduals_to_combine.size() != 0) {
        result.addAll(combine(inviduals_to_combine));
      }
    }

    /**
     * strategy specify if rest of individuals in population are added to the
     * result*
     */
    result =
        crossover_strategy.addUnselected(result, combine_parent_selector
            .getUnselectedIndividuals());

    return new EvPopulation<T>(result);
  }


  /**
   * Combines given individuals to create some new, maybe better ones. This
   * function is called (usually) several times when apply() function of the
   * operator has been called. Each call produces some individuals, results from
   * all those calls are returned by apply() method.<br>
   * The number of combine() calls per one apply() is defined by
   * EvCrossoverApplyStrategy object, parent selected in each call are defined
   * by EvCombineParentSelector object.
   * 
   * @param parents individuals - list must contain numbers of individuals
   *        specified by arity() function
   * @return new individuals - list containg number of individual defined by
   *         combineResultSize() function
   */
  public abstract List<T> combine(List<T> parents);


  /**
   * Returns how many individuals are needed as argument for combine()
   * 
   * @return
   */
  public abstract int arity();


  /**
   * Returns how many individuals are created by single call of combine()
   * 
   * @return
   */
  public abstract int combineResultSize();


  /**
   * Sets the object responsible for selecting parent during each combine()
   * call.<br>
   * Uses EvSimpleCombineSelector by default.
   * 
   * @param combine_parent_selector
   */
  public void setCombineParentSelector(
      EvCombineParentSelector<T> combine_parent_selector) {
    this.combine_parent_selector = combine_parent_selector;
  }


  /**
   * Sets the object responsible which tells how many times call combine() per
   * each call of the operator.
   * 
   * @param crossover_strategy
   */
  public void setCrossoverStrategy(EvCrossoverApplyStrategy crossover_strategy) {
    this.crossover_strategy = crossover_strategy;
  }

}