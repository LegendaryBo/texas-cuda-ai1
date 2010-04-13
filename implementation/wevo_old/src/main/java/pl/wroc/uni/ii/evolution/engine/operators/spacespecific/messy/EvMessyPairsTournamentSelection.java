package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvPairsTournamentSelection;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Messy pairs tournament sequenced selection. Like a pairs tournament sequenced
 * selection, but provided additional support for tie breaking and thresholding.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 * @param <T> type of EvMessyIndividual
 */
public class EvMessyPairsTournamentSelection<T extends EvMessyIndividual>
    extends EvPairsTournamentSelection<T> {

  protected boolean thresholding;

  protected boolean tie_breaking;


  /**
   * Constructor, creates the tournament with specified parameters.
   * 
   * @param winners_number number of winners
   * @param thresholding there are compared only individuals which have common
   *        specified genes number is at least the certain number
   * @param tie_breaking with this option, if individuals in the pair has the
   *        same fitness, then there is selected shorter individual.
   */
  public EvMessyPairsTournamentSelection(int winners_number,
      boolean thresholding, boolean tie_breaking) {
    super(winners_number);
    this.tie_breaking = tie_breaking;
    this.thresholding = thresholding;
  }


  @Override
  @SuppressWarnings("unchecked")
  public int getNextIndex() {
    if (!thresholding)
      return super.getNextIndex();

    // Check end of permutation
    if (position >= population.size() - 1) {
      permutation = EvRandomizer.INSTANCE.nextPermutation(population.size());
      position = 0;
    }

    /*
     * Search for a partner, if it will not be found, then the partner is next
     * individual and first individual is the winner.
     */
    int result = permutation[position];

    // The range of search is the genotype length
    int thresholding_search_range = population.get(0).getGenotypeLength();

    for (int k = 1; k <= thresholding_search_range; k++) {
      // Rewind if end of permutation exceeded
      int partner = position + k;
      if (partner >= population.size())
        partner -= population.size();

      // Check the common genes with partner
      int threshold =
          population.get(position).getExpressedGenesNumber()
              * population.get(partner).getExpressedGenesNumber()
              / population.get(position).getGenotypeLength();

      if (population.get(permutation[position]).getCommonExpressedGenesNumber(
          population.get(permutation[partner])) > threshold) {
        // Found a partner
        if (fight(population.get(permutation[position]), population
            .get(permutation[partner])))
          result = permutation[position];
        else
          result = permutation[partner];

        // Swap with next individual
        int swap = permutation[position + 1];
        permutation[position] = permutation[partner];
        permutation[partner] = swap;

        break;
      }
    }
    position += 2;

    return result;
  }


  /**
   * The winner is individual which has better objective function value, if
   * there is a tie and tie_breaking is turned on, then wins individual with
   * shorter chromosome.
   * 
   * @param i1 first individual
   * @param i1 second individual
   * @returns result true iff the first individual is the better, if there is a
   *          tie and tie breaking is used, wins the shorter individual, else
   *          first individual is the winner.
   */
  @Override
  protected boolean fight(EvMessyIndividual i1, EvMessyIndividual i2) {
    double v1 = i1.getObjectiveFunctionValue();
    double v2 = i2.getObjectiveFunctionValue();
    if (v1 > v2)
      return true;
    else if (v1 < v2)
      return false;
    else if (tie_breaking)
      return i1.getChromosomeLength() <= i2.getChromosomeLength();
    else
      return true;
  }

}