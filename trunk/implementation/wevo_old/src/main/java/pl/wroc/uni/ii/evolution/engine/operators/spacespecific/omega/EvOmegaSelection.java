package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.omega;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Selection for Omega algorithm
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowsi (m4linka@gmail.com)
 */
public class EvOmegaSelection extends EvSelection<EvOmegaIndividual> {

  private final int output_population_length;

  private final int number_of_candidates;


  /**
   * Constructor
   * 
   * @param output_population_length number of selections to perform
   * @param number_of_candidates number of picked candidates
   */
  public EvOmegaSelection(int output_population_length, int number_of_candidates) {

    this.output_population_length = output_population_length;
    this.number_of_candidates = number_of_candidates;
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<EvOmegaIndividual> population) {
    List<Integer> result = new ArrayList<Integer>(output_population_length);

    for (int i = 0; i < output_population_length; i++) {
      int pop_size = population.size();
      final int ind_index = EvRandomizer.INSTANCE.nextInt(pop_size);
      final EvOmegaIndividual ind = population.get(ind_index);

      int candidates_indexes[] = new int[number_of_candidates];

      for (int pick = 0; pick < number_of_candidates; pick++) {
        candidates_indexes[pick] = EvRandomizer.INSTANCE.nextInt(pop_size);
      }

      int best_candidate_index = -1;
      for (int j = 0; j < number_of_candidates; j++) {
        EvOmegaIndividual candidate = population.get(candidates_indexes[j]);

        // required shared genes number
        final int theta =
            (int) (ind.getChromosomeLength() * candidate.getChromosomeLength() / ind
                .getGenotypeLength());

        if (ind.getCommonExpressedGenesNumber(candidate) >= theta) {
          best_candidate_index = candidates_indexes[j];
          break;
        }
      }

      if (best_candidate_index >= 0) {
        // we have the competition
        EvOmegaIndividual best_candidate = population.get(best_candidate_index);
        if (ind.compareTo(best_candidate) < 0) {
          result.add(best_candidate_index);
        } else {
          result.add(ind_index);
        }
      } else {
        result.add(ind_index);
      }
    }
    return result;
  }
}
