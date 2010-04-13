package pl.wroc.uni.ii.evolution.engine.prototype.operators;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * An abstract class for all selection operators. <br>
 * A selection operator returns some multiset of given individuals.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @param <T> - type of individuals the selection works on
 */
public abstract class EvSelection<T extends EvIndividual> implements
    EvOperator<T> {

  @SuppressWarnings("unchecked")
  /**
   * This function builds new population from list of indexes created by
   * function getIndexes. If you want to write your own operator, you need to
   * overwrite function "List<Integer> getIndexes(EvPopulation<T> population)"
   * which return list of indexes of individuals selected by the operator.
   */
  public EvPopulation<T> apply(EvPopulation<T> population) {

    EvPopulation<T> result = new EvPopulation<T>();
    List<Integer> individuals_indexes = getIndexes(population);
    int population_size = individuals_indexes.size();
    Integer[] indexes_array =
        individuals_indexes.toArray(new Integer[population_size]);

    for (int i = 0; i < population_size; i++) {
      result.add((T) population.get(indexes_array[i]).clone());
    }

    return result;
  }


  @SuppressWarnings("unchecked")
  public static <T extends EvIndividual> EvPopulation<T> apply(
      EvPopulation<T> population, List<Integer> indexes) {
    EvPopulation<T> result = new EvPopulation<T>();

    int indexes_size = indexes.size();
    for (int i = 0; i < indexes_size; i++) {
      result.add((T) population.get(i).clone());
    }

    return result;
  }


  /**
   * Returns indexes of unselected idividuals
   * 
   * @param indexes indexes of selected individuals
   * @param population_size size of population
   * @return list of Integer
   */
  public static List<Integer> getUnselected(List<Integer> indexes,
      int population_size) {

    boolean[] tmp = new boolean[population_size];

    for (int i = 0; i < tmp.length; i++) {
      tmp[i] = false;
    }

    for (int i = 0; i < indexes.size(); i++) {
      tmp[indexes.get(i)] = true;
    }

    List<Integer> result = new ArrayList<Integer>();
    for (int i = 0; i < tmp.length; i++) {

      if (!tmp[i]) {
        result.add(i);
      }
    }
    return result;
  }


  /**
   * Returns indexes of selected individuals. If there aren't inviduals selected
   * a selection should return empty list
   * 
   * @param population a given population of individuals
   * @return indexes
   */
  public abstract List<Integer> getIndexes(EvPopulation<T> population);

}
