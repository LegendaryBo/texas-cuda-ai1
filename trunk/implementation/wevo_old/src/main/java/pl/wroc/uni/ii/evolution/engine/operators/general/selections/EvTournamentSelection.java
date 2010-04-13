package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A tournament selection with return. It randomly chooses individuals group
 * with <code> size = tournament_size </code> and add best
 * <code> winner_numbers </code> individual to result until result and given
 * population sizes are not equal.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> Type of the individual.
 */
public class EvTournamentSelection<T extends EvIndividual> extends
    EvSelection<T> {

  /** Random selection used in tournament. */
  private final EvRandomSelection<T> random_selection;

  /** Best selection that creates tournament. */
  private final EvKBestSelection<T> best_selection;

  /** Random + best selection used in single tournament. */
  private final EvSelection<T> single_tournament_selection;

  /** Number of individuals in result. */
  private int result_size = -1;


  /**
   * Creates tournament selection operator with given tournament size and number
   * of winners of each tournament. Final size of the population will be equal
   * to the size of source population.
   * 
   * @param tournament_size Size of the tournament.
   * @param winner_numbers Number of winners in each tournament.
   */
  public EvTournamentSelection(final int tournament_size,
      final int winner_numbers) {
    if (tournament_size < winner_numbers) {
      throw new IllegalArgumentException(
          "tournament_size must be greater than winner_number");
    }

    random_selection = new EvRandomSelection<T>(tournament_size, true);
    best_selection = new EvKBestSelection<T>(winner_numbers);
    single_tournament_selection =
        new EvTwoSelectionComposition<T>(random_selection, best_selection);
  }


  /**
   * Creates tournament selection operator with given tournament size and number
   * of winners of each tournament.
   * 
   * @param tournament_size Size of the tournament.
   * @param winner_numbers Number of winners in a single tournament.
   * @param count Number of individuals in resulting population.
   */
  public EvTournamentSelection(final int tournament_size,
      final int winner_numbers, final int count) {
    if (tournament_size < winner_numbers) {
      throw new IllegalArgumentException(
          "tournament_size must be greater than winner_number");
    }

    random_selection = new EvRandomSelection<T>(tournament_size, true);
    best_selection = new EvKBestSelection<T>(winner_numbers);
    single_tournament_selection =
        new EvTwoSelectionComposition<T>(random_selection, best_selection);
    this.result_size = count;
  }


  /** {@inheritDoc} */
  @Override
  public List<Integer> getIndexes(final EvPopulation<T> population) {
    int desired_size = population.size();
    if (result_size != -1) {
      desired_size = result_size;
    }

    List<Integer> result = new ArrayList<Integer>();

    while (result.size() < desired_size) {
      List<Integer> tmp = single_tournament_selection.getIndexes(population);
      result.addAll(tmp);
    }

    while (result.size() > desired_size) {
      result.remove(EvRandomizer.INSTANCE.nextInt(result.size()));
    }
    return result;
  }

}