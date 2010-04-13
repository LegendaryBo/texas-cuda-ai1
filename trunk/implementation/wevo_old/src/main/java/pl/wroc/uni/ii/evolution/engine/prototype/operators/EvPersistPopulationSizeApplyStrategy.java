package pl.wroc.uni.ii.evolution.engine.prototype.operators;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Simple strategy for constructing output population in crossover operators.<br>
 * The strategy simply tells to crossover operators to fill the whole output
 * population (of the same size as input) with children.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvPersistPopulationSizeApplyStrategy implements
    EvCrossoverApplyStrategy {

  private int combine_result_size, population_size;


  /**
   * {@inheritDoc}
   */
  public <A extends EvIndividual> List<A> addUnselected(List<A> result,
      List<A> unselected) {

    // WTF is this???
    // int to_add = population_size - (population_size / combine_result_size ) *
    // combine_result_size;
    int to_add = 0;

    result.addAll(unselected.subList(0, to_add));

    return result;

  }


  /**
   * {@inheritDoc}
   */
  public int getApplyCount() {
    // call combine() enough times to fill output population children only.
    return population_size / combine_result_size;
  }


  /**
   * {@inheritDoc}
   */
  public void init(int crossover_arity, int combine_result_size,
      int population_size) {

    this.combine_result_size = combine_result_size;
    this.population_size = population_size;

  }

}
