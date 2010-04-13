package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * Standard block selection. It returns population of the same size as input
 * population. It removes M worst individuals and copy M best individuals.
 * 
 * @author: Piotr Baraniak, Marek Chruœciel
 */
public class EvBlockSelection<T extends EvIndividual> extends EvSelection<T> {

  /**
   * A class representing 'comparable' individual
   */
  static private class Element implements Comparable<Element> {

    public double value;

    public int index;


    /**
     * @param index of the individual
     * @param value of the objective function
     */
    public Element(int index, double value) {
      this.index = index;
      this.value = value;
    }


    public int compareTo(Element o) {
      if (o.value == this.value) {
        return 0;
      }
      if (o.value < this.value) {
        return 1;
      }
      return -1;
    }
  }

  private final int M;


  /**
   * @param M - number of worst individuals which we want remove
   */
  public EvBlockSelection(int M) {
    this.M = M;
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<T> population) {

    if (population.size() / 2 < M) {
      throw new IllegalStateException(
          "EvBlockSelection can't be apply on population smaller than having 2 M individuals");
    }

    List<Integer> result = new ArrayList<Integer>();
    Element[] elements = new Element[population.size()];

    for (int i = 0; i < population.size(); i++) {
      elements[i] =
          new Element(i, population.get(i).getObjectiveFunctionValue());
    }

    // sorting by objective function value
    Arrays.sort(elements);

    /* Copying best individuals to result */
    for (int i = 0; i < M; i++) {
      result.add(elements[elements.length - 1 - i].index);
    }

    /* Copying rest individuals to result */
    for (int i = M; i < population.size(); i++) {
      result.add(elements[i].index);
    }

    return result;

  }
}
