package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.combineselector.EvSimpleCombineSelector;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCombineParentSelector;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Cut and splice operators together for EvMessyIndividual. This combines 2
 * individuals, cuts each of them with cut probability and splices created parts
 * with splice probability. Each combining produces 1-4 children from 2 parents.
 * CutSplice has own crossover strategy, combining individuals while the new
 * population is not filled to the input population size. CombineParentSelector
 * for CutSplice is called to give 2 * population size number of parents lists
 * (which is maximal number of parents that can be needed), but the real count
 * can be less since desired children number is reached.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 * @param <T> type of EvMessyIndividual on which the operator works
 */
public class EvMessyCutSpliceOperator<T extends EvMessyIndividual> implements
    EvOperator<T> {

  private EvCombineParentSelector<T> combine_parent_selector;

  // probability of cut an individual
  private double cut_probability;

  // probability of splice 2 individuals
  private double splice_probability;


  /**
   * Constructor, creates MessyCutSplice with given parameters.
   * 
   * @param cut_probability probability of cut an individual
   * @param splice_probabolity probability of splice 2 invdividuals
   */
  public EvMessyCutSpliceOperator(double cut_probability,
      double splice_probability) {
    if (cut_probability < 0.0 && cut_probability > 1.0)
      throw new IllegalArgumentException(
          "Cut probability must be a double in [0,1]");
    if (splice_probability < 0.0 && splice_probability > 1.0)
      throw new IllegalArgumentException(
          "Splice probability must be a double in [0,1]");

    combine_parent_selector = new EvSimpleCombineSelector<T>();
    this.cut_probability = cut_probability;
    this.splice_probability = splice_probability;
  }


  /**
   * Sets a CombineParentSelector.
   * 
   * @param combine_parent_selector new combine parent selector to set
   */
  public void setCombineParentSelector(
      EvCombineParentSelector<T> combine_parent_selector) {
    if (combine_parent_selector == null)
      throw new IllegalArgumentException(
          "CombinedParentSelector cannot be null");

    this.combine_parent_selector = combine_parent_selector;
  }


  /**
   * Applies the operator. Creates new population with minimal population size
   * from individuals after cut and splice operations.
   * 
   * @param population population on which operator works
   * @return population new population created by applying the operator
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {

    int population_size = population.size();

    combine_parent_selector.init(population, 2, population_size);

    EvPopulation<T> result = new EvPopulation<T>(population_size + 3);

    // Generate children since population is not filled
    while (result.size() < population_size)
      combine(combine_parent_selector.getNextParents(), result);

    // Remove overfull
    while (result.size() > population_size)
      result.remove(result.size() - 1);

    return result;
  }


  /**
   * Combines 2 parents and performs cut and splice to create children. Created
   * children number is in [1,4].
   * 
   * @param parents list of 2 parents
   * @return list of children
   */
  public void combine(List<T> parents, List<T> result) {
    assert (parents.size() == 2);

    List<T> list1 = Cut(parents.get(0));// A B
    List<T> list2 = Cut(parents.get(1));// C D

    /*
     * insert flipped list2 in the middle of list1 so there will be A D C B
     */
    if (list2.size() < 2) {
      list1.add(1, list2.get(0));
    } else {
      list1.add(1, list2.get(0));
      list1.add(1, list2.get(1));
    }

    Splice(list1, result);
  }


  // Cuts individual and return a list with the parts of it
  @SuppressWarnings("unchecked")
  private List<T> Cut(T individual) {
    ArrayList<T> list = new ArrayList<T>(2);
    int length = individual.getChromosomeLength();

    if (EvRandomizer.INSTANCE.nextDouble() < (length - 1) * cut_probability) {

      int cut_place = EvRandomizer.INSTANCE.nextInt(length - 1);

      T child1 = (T) individual.clone();
      child1.setChromosome(individual.getGenesList(0, cut_place), individual
          .getAllelesList(0, cut_place));
      T child2 = (T) individual.clone();
      child2.setChromosome(individual.getGenesList(cut_place, length),
          individual.getAllelesList(cut_place, length));
      list.add(child1);
      list.add(child2);

    } else
      list.add((T) individual.clone());

    return list;
  }


  // Append spliced individuals from list to result
  @SuppressWarnings("unchecked")
  private void Splice(List<T> list, List<T> result) {
    int size = list.size();

    int i = 0;
    while (i < size - 1)
      if (EvRandomizer.INSTANCE.nextDouble() < splice_probability) {
        T child = list.get(i);
        child.addAlleles(list.get(i + 1).getGenes(), list.get(i + 1)
            .getAlleles());

        result.add(child);
        i += 2;
      } else {
        result.add(list.get(i));
        i++;
      }
    if (i < size)
      result.add(list.get(i));
  }

}