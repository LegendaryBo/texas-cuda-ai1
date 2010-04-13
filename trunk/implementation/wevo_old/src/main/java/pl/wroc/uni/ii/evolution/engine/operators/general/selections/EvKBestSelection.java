package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSelection;

/**
 * Selects k best individuals for the population. An example of EvSelection
 * operator. Really nothing fancy.
 * 
 * @author Marcin Brodziak, Tomasz Kozakiewicz
 */
public class EvKBestSelection<T extends EvIndividual> extends EvSelection<T> {

  static private class Element implements Comparable<Element> {

    public double value;

    public int index;


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

  /**
   * A number of individuals preserved in generated population.
   */
  private int k;


  /**
   * Constructor with a single parameter that represents a number of individuals
   * preserved in target population.
   * 
   * @param k how many best individuals are selected
   */
  public EvKBestSelection(int k) {
    this.k = k;
  }


  @Override
  public List<Integer> getIndexes(EvPopulation<T> population) {

    List<Integer> result = new ArrayList<Integer>();
    Element[] elements = new Element[population.size()];

    int population_size = population.size();
    for (int i = 0; i < population_size; i++) {
      elements[i] =
          new Element(i, population.get(i).getObjectiveFunctionValue());
    }

    Arrays.sort(elements);

    /* Copying best individuals to result */
    for (int i = 0; i < k; i++) {
      result.add(elements[elements.length - 1 - i].index);
    }

    return result;
  }
}
