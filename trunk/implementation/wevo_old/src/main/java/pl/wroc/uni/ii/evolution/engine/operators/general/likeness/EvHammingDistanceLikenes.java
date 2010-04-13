package pl.wroc.uni.ii.evolution.engine.operators.general.likeness;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;

/**
 * 
 * Class used to select the most similar individual from the group to
 * the one individual specified.<br>
 * Similarity between 2 individuals is rated according to hamming distance
 * function. Rating is equal to the number of different genes in the 
 * individuals.
 * 
 * @see http://en.wikipedia.org/wiki/Hamming_distance
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 * @param <T> - type of individual
 */
public class EvHammingDistanceLikenes<T extends EvKnaryIndividual>
    implements EvLikenes<T> {

  /**
   * {@inheritDoc}
   */
  public T getSimilar(final EvPopulation<T> candidates, final T pattern) {

    if (candidates == null || candidates.size() == 0) {
      throw new IllegalStateException("No candidates!");
    }
    
    int best_matching_ind = 0;
    int min_value = Integer.MAX_VALUE; 
    
    // compare patter which each candidate
    for (int i = 0; i < candidates.size(); i++) {
      int current_value = getLikenes(pattern, candidates.get(i));

      if (current_value < min_value) {
        best_matching_ind = i;
        
        min_value = current_value;
      }
    }
    
    return candidates.get(best_matching_ind);
  }

  /**
   * 
   * Rate similarity between 2 individuals.
   * Returns number of different genes.
   * 
   * @param a - first individual.
   * @param b - second individual.
   * @return number of different genes.
   */
  private int getLikenes(final T a, final T b) {
    
    int likenes = 0;
    
    for (int i = 0; i < a.getDimension(); i++) {
      
      if (a.getGene(i) != b.getGene(i)) {
        likenes++;
      }
    }
    return likenes;
  }

}
