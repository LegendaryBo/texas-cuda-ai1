package pl.wroc.uni.ii.evolution.experimental.decisiontree;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.experimental.EvTreeIndividual;
import pl.wroc.uni.ii.evolution.utils.EvIRandomizer;

/**
 * A crossover operator for tree individuals. It takes population of two
 * individuals and returns a new population of two children. Children are new
 * individuals created in the following manner: A node is drawed at random from
 * each parent. These nodes define the subtrees that will be swaped between the
 * two.
 * 
 * @author Kamil Dworakowski
 * @param <T> type of individual the operator works on
 */
public class EvTreeCrossover<T extends EvTreeIndividual<T>> implements
    EvOperator<T> {

  private EvIRandomizer randomizer;


  public EvTreeCrossover(EvIRandomizer randomizer) {
    this.randomizer = randomizer;
  }


  @SuppressWarnings("unchecked")
  public EvPopulation<T> apply(EvPopulation<T> population) {
    assert population.size() == 2;

    T father = population.get(0);
    T mother = population.get(1);

    T subtree_of_fahter = father.randomDescendant(randomizer);
    T subtree_of_mother = mother.randomDescendant(randomizer);

    EvPopulation<T> result_pop = new EvPopulation<T>();

    result_pop.add((T) father.replace(subtree_of_fahter, subtree_of_mother)
        .clone());
    result_pop.add((T) mother.replace(subtree_of_mother, subtree_of_fahter)
        .clone());

    return result_pop;
  }
}
