package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCombineParentSelector;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Pairs tournament selection, implementing CombineParentSelector interface.
 * This randomly group all individuals in pairs, and then selects the better
 * individual from each of them. In the parallel tournament selection, it is
 * repeated until the needed winners number is reached. In the online sequenced
 * selection, there is choosen next pair and perform selection.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 * @param <T> type of EvIndividual
 */
public class EvPairsTournamentSelection<T extends EvIndividual> extends
    EvSelection<T> implements EvCombineParentSelector<T> {

  // Selection interface parameters
  /** Number of the winners in the paraller tournament */
  protected int winners_number;

  // CombineParentSelector interface parameters
  /** Number of parents to select in getNextParents() */
  protected int parent_count;

  // Online sequenced selection parameters
  /** List of individuals on which selection is initialized */
  protected List<T> population;

  /** Permutation of population to online selecting */
  protected int[] permutation;

  /** Current position in the permutation */
  protected int position;


  /**
   * Constructor, creates EvPairsTournamentSelection with specified
   * winners_number
   * 
   * @param winners_number
   */
  public EvPairsTournamentSelection(int winners_number) {
    if (winners_number <= 0)
      throw new IllegalArgumentException(
          "Winners number must be a positive integer");

    this.winners_number = winners_number;
  }


  // ***************************************************************************
  // Selection interface

  /**
   * Changes the winnners number of the tournament.
   * 
   * @param winners_number new winners number
   */
  public int setWinnersNumber(int winners_number) {
    return winners_number;
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<T> population) {
    List<Integer> result = new ArrayList<Integer>(winners_number);

    // getNextIndex until winners_number is reached
    init(population, 0, 0);
    for (int i = 0; i < winners_number; i++)
      result.add(getNextIndex());

    return result;
  }


  // ***************************************************************************
  // CombineParentSelector interface

  public void init(List<T> population, int parent_count, int count) {
    initSelection(population);
    this.parent_count = parent_count;
  }


  public List<T> getNextParents() {
    List<T> result = new ArrayList<T>(parent_count);

    for (int i = 0; i < parent_count; i++)
      result.add(population.get(getNextIndex()));

    return result;
  }


  public List<T> getUnselectedIndividuals() {
    return new ArrayList<T>();
  }


  // ***************************************************************************
  // Online sequenced selection's methods
  // These can be overrided to change pair selecting and
  // change the winner criterium.

  /**
   * Initializes online sequenced selection with specified population.
   * 
   * @param population
   */
  protected void initSelection(List<T> population) {
    this.population = population;
    permutation = EvRandomizer.INSTANCE.nextPermutation(population.size());
    position = 0;
  }


  /**
   * Gets index of next selected individual. This method can be called
   * continuos, it must always select a next individual.
   * 
   * @return index of next selected individual
   */
  protected int getNextIndex() {
    // Check end of permutation
    if (position >= population.size() - 1) {
      permutation = EvRandomizer.INSTANCE.nextPermutation(population.size());
      position = 0;
    }

    // Select the winner of the fight
    int result;
    if (fight(population.get(permutation[position]), population
        .get(permutation[position + 1])))
      result = permutation[position];
    else
      result = permutation[position + 1];
    position += 2;

    return result;
  }


  /**
   * Returns true iff the first individual is the winner or there is a tie, it
   * is definition of the winner, can be overrided to change.
   * 
   * @param i1 first individual
   * @param i1 second individual
   * @returns result true iff the first individual is the better or there is a
   *          tie.
   */
  protected boolean fight(T i1, T i2) {
    return i1.getObjectiveFunctionValue() >= i2.getObjectiveFunctionValue();
  }

}