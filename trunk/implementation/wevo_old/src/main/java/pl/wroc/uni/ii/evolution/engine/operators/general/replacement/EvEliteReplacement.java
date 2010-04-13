package pl.wroc.uni.ii.evolution.engine.operators.general.replacement;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;

/**
 * Replacement that allows a small number of parents, so called elite, to
 * survive to next iteration. The bigger the size of the elite the faster the
 * algorithms converges (gets trapped in local maximums more easiely). The lower
 * the size of elite the less performence it has. Optimally, elite should be
 * small, 2 for examle.
 * 
 * @author Kamil Dworakowski
 * @param <T>
 */
public class EvEliteReplacement<T extends EvIndividual> implements
    EvReplacement<T> {

  private int elite_size, output_population_size;


  public EvEliteReplacement(int output_population_size, int elite_size) {
    this.elite_size = elite_size;
    this.output_population_size = output_population_size;
  }


  /**
   * Default value for elite size is 2.
   */
  public EvEliteReplacement(int output_population_size) {
    this(output_population_size, 2);
  }


  public EvPopulation<T> apply(EvPopulation<T> parents, EvPopulation<T> children) {
    // cloning children population
    EvPopulation<T> temp = new EvPopulation<T>(children);
    parents.sort();
    // adding elite individuals
    System.out.println(parents.get(parents.size() - 1).getObjectiveFunctionValue());
    for (int i = 0; i < elite_size; i++) {
      temp.add(parents.get(parents.size() - i - 1));
    }
    // deleting worsts individuals
    return temp.kBest(output_population_size);
  }

}
