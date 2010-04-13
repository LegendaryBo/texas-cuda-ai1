package pl.wroc.uni.ii.evolution.engine.prototype.operators;

import java.util.List;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * A very simple apply crossover strategy. For operator with arity <i> A </i>
 * and population size <i> N </i> the stategy specify that crossover should call
 * combine on parents, choosed by combine selection, <i> N / A </i> times.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the strategy works on
 */
public class EvSimpleApplyStrategy implements EvCrossoverApplyStrategy {

  private int crossover_arity, population_size;


  public <A extends EvIndividual> List<A> addUnselected(List<A> result,
      List<A> unselected) {
    result.addAll(unselected);
    return result;
  }


  public int getApplyCount() {
    return population_size / crossover_arity;
  }


  public void init(int crossover_arity, int combine_result_size,
      int population_size) {
    this.crossover_arity = crossover_arity;
    this.population_size = population_size;
  }
}
