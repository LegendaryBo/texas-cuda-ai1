package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Selects k best individuals for the population. An example of
 * EvolutionaryOperator. Really nothing fancy.
 * 
 * @author Marcin Brodziak, Tomasz Kozakiewiczs
 */
public class EvRandomSelection<T extends EvIndividual> extends EvSelection<T> {
  /**
   * A number of individuals in the result of the EvTotalRandomSelection.
   */
  private int count;

  private boolean allow_duplicated_indexes;


  /**
   * Constructor with a single parameter that represents a number of individuals
   * preserved in a result population.
   * 
   * @param count how many individuals are selected
   * @parem allow_duplicated_indexes allow choose many time the same individual
   */
  public EvRandomSelection(int count, boolean allow_duplicated_indexes) {
    this.count = count;
    this.allow_duplicated_indexes = allow_duplicated_indexes;
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<T> population) {
    List<Integer> result = new ArrayList<Integer>(count);

    if (allow_duplicated_indexes) {

      for (int i = 0; i < count; i++) {
        result.add(EvRandomizer.INSTANCE.nextInt(population.size()));
      }
      return result;

    } else {
      boolean[] indexes =
          EvRandomizer.INSTANCE.nextBooleanList(population.size(), count);
      for (int i = 0; i < population.size(); i++) {
        if (indexes[i]) {
          result.add(i);
        }
      }
      return result;
    }
  }
}
