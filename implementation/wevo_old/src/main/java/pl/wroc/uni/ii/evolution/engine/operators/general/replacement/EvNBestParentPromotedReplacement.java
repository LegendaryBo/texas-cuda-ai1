package pl.wroc.uni.ii.evolution.engine.operators.general.replacement;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;

/**
 * A default implementation of replacement operator. n best parents are being
 * promoted to the next population. The resulting population has the same number
 * of elements as the parents population.
 * 
 * @author Kamil Dworakowski
 * @param <T>
 */
public class EvNBestParentPromotedReplacement<T extends EvIndividual>
    implements EvReplacement<T> {

  private int how_many_parents_to_promote;


  public EvNBestParentPromotedReplacement(int how_many_parents_to_promote) {
    assert how_many_parents_to_promote >= 0;
    this.how_many_parents_to_promote = how_many_parents_to_promote;
  }


  public EvPopulation<T> apply(EvPopulation<T> parents, EvPopulation<T> children) {
    assert how_many_parents_to_promote <= parents.size();

    EvPopulation<T> new_population = new EvPopulation<T>(parents.size());
    new_population
        .addAll(kBestIndividuals(parents, how_many_parents_to_promote));
    int how_many_more = parents.size() - how_many_parents_to_promote;
    new_population.addAll(kBestIndividuals(children, how_many_more));

    return new_population;
  }


  // TODO KBestSelection makes clones and that is unncessary waste
  // implement KBestIndividuals method on Population
  private EvPopulation<T> kBestIndividuals(EvPopulation<T> children,
      int how_many_more) {
    EvKBestSelection<T> k_selection_children =
        new EvKBestSelection<T>(how_many_more);

    EvPopulation<T> apply = k_selection_children.apply(children);
    return apply;
  }

}
