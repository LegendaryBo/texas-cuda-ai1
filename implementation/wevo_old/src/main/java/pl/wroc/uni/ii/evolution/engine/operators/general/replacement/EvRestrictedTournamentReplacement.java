package pl.wroc.uni.ii.evolution.engine.operators.general.replacement;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.likeness.EvLikenes;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;

/**
 * 
 * Implementation of restricted tournament replacement operator.<br>
 * <br>
 * It replaces individuals in the following way:<br>
 * - take individual x from children population X.<br>
 * - select the most similar individual y from subset parent population Y.<br>
 * - replace x with y in population Y if x > y (otherwise do nothing). <br>
 * - repeat for all individuals in X.<br>
 * - return population Y.<br>
 * <br>
 * 
 * 
 * @author Kacper Gorski (admin@34all.org)
 * 
 * @param <T> - type of individual.
 *  
 *
 */
public class EvRestrictedTournamentReplacement<T extends EvIndividual>
    implements EvReplacement<T> {

  
  /**
   * Function of likenes. It says how likely is one individual to another.
   */
  private EvLikenes<T> likenes = null;
  
  /**
   * Size of subset of population Y.
   */
  private int tournament_size = 0;
  
  /**
   * Used to select subset of population Y.
   */
  private EvRandomSelection<T> random_selection = null;
      
  
  /**
   * @see 
   * Creates restricted tournament operator. It matches individuals
   * using given likenes function. Each new individual compete in 
   * tournament of specified size with individual that is most
   * similar to it.
   * 
   * @param tournament_size_ - size of single tournament for each new
   * individual. Recommended same size as nummber of genes in individuals.
   * @param likenes_ - explains the way in which individuals likenes is rated.
   */
  public EvRestrictedTournamentReplacement(
      final int tournament_size_, final EvLikenes<T> likenes_) {
    likenes = likenes_;
    tournament_size = tournament_size_;
    random_selection = new EvRandomSelection<T>(tournament_size, true);
  }
  
  /**
   * {@inheritDoc}
   */
  public EvPopulation<T> apply(
      final EvPopulation<T> parents, final EvPopulation<T> children) {   

    for (int i = 0; i < children.size(); i++) {
      
      T candidate = children.get(i);
      T most_similar =  likenes.getSimilar(
          random_selection.apply(parents), children.get(i));

      if (most_similar.getObjectiveFunctionValue() 
          < candidate.getObjectiveFunctionValue()) {
        if (parents.remove(most_similar))
          parents.add(candidate);
      }
    }
    
    return parents;
  }

}
