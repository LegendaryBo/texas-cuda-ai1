package pl.wroc.uni.ii.evolution.engine.operators.general.replacement;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;

/**
 * A default implementation of replacement operator. N best individuals from
 * union of both populations are returned.
 * 
 * @author Kamil Dworakowski
 * @param <T>
 */
public class EvBestFromUnionReplacement<T extends EvIndividual> implements
    EvReplacement<T> {

  private int size = -1;


  /**
   * This constructor does not set the size of population to return. It will
   * return a population with the same size as the input parrents population
   */
  public EvBestFromUnionReplacement() {
    super();
  }


  /**
   * @param size how many individuals to put into output population
   */
  public EvBestFromUnionReplacement(int size) {
    super();
    this.size = size;
  }


  public EvPopulation<T> apply(EvPopulation<T> parents, EvPopulation<T> children) {
    EvPopulation<T> new_population = new EvPopulation<T>();
    new_population.addAll(parents);
    new_population.addAll(children);

    if (size != -1)
      return new_population.kBest(size);
    else
      return new_population.kBest(parents.size());
  }
}
