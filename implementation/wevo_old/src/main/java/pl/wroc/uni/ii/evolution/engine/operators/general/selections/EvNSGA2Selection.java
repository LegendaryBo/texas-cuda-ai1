package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.multiobjective.EvCrowdingDistance;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.multiobjective.EvParetoFrontsRank;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Selection for Non-Dominated Sorting Genetic Algorithm II.
 * Selects better individuals for k random pairs of individuals
 * from the population.
 * Individual is better if it has bigger rank value (rank based on
 * Pareto fronts) or if compared individuals ranks are equal
 * we look at crowding distance.
 * 
 * @author Adam Palka
 * @param <T> - type of individuals
 */
public class EvNSGA2Selection<T extends EvIndividual> extends EvSelection<T> {

  /** A number of individuals preserved in generated population. */
  private int k;

  /**
   * Constructor with a single parameter that represents a number of individuals
   * preserved in target population.
   * 
   * @param i -individuals selected
   */
  public EvNSGA2Selection(final int i) {
    this.k = i;
  }

  /**
   * Return indexes of selected individuals.
   * @param population from which operator will select individuals
   * @return indexes
   */
  @Override
  public List<Integer> getIndexes(final EvPopulation<T> population) {

    List<Integer> result = new ArrayList<Integer>();

    EvParetoFrontsRank<T> rank = new EvParetoFrontsRank<T>(population);
    EvCrowdingDistance<T> cDistance = null;
    for (int i = 0; i < k; i++) {
      int[] number = new int[2];
      number[0] = EvRandomizer.INSTANCE.nextInt(population.size());
      number[1] = EvRandomizer.INSTANCE.nextInt(population.size());
      if (rank.getRank(number[0]) > rank.getRank(number[1])) {
        result.add(number[0]);
      } else if (rank.getRank(number[0]) < rank.getRank(number[1])) {
        result.add(number[1]);
      } else {
        if (cDistance == null) {
          cDistance = new EvCrowdingDistance<T>(population);
        }
        if (cDistance.getCrowdingDistance(number[0])
            < cDistance.getCrowdingDistance(number[1])) {
          result.add(number[0]);
        } else if (cDistance.getCrowdingDistance(number[0])
            > cDistance.getCrowdingDistance(number[1])) {
          result.add(number[1]);
        } else {
          result.add(number[EvRandomizer.INSTANCE.nextInt(2)]);
        }
      }
    }
    return result;
  }
}
