package pl.wroc.uni.ii.evolution.engine.operators.general.selections.multiobjective;


import java.util.Arrays;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Class use to compute crowding distance for population.
 * An estimate of the density of solutions surrounding a particular point 
 * in the population we take the average distance of the two points
 * on either side of this point along each of the objectives.
 * If crowding distance equals Double.MAX_VALUE it means
 * his crowding distance is infinitive.
 * 
 * @author Adam Palka
 * @param <T> - type of individuals
 */
public class EvCrowdingDistance<T extends EvIndividual> {

  /** Population for which we want to compute crowding distance. */
  private EvPopulation<T> population;
  /** Table witch crowding distances of each individual. */
  private double[] crowdingDistance;
  
  /** Class used to remember indexes of individuals and sorting. */
  private static class Element implements Comparable<Element> {

    /** Individual number. */
    private int index;
    /** Value of objective function used for sorting. */
    private double value;
    /** Distance. */
    private double distance;

    /**
     * Constructor.
     * @param i -individual number
     */
    public Element(final int i) {
      this.index = i;
      this.distance = 0;
    }

    /** Compares two elements.
     * @param o -individual
     * @return result of comparison
     */
    public int compareTo(final Element o) {
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
   * Constructor which initialize computing of crowding distance.
   * @param pop -population
   */
  public EvCrowdingDistance(final EvPopulation<T> pop) {
    this.population = pop;
    this.crowdingDistance = new double[population.size()];
    computeCrowdingDistance();
  }

  /**
   * Getter.
   * @param k -number of individual.
   * @return crowding distance value of individual k from population.
   */
  public double getCrowdingDistance(final int k) {
    return crowdingDistance[k];
  }
  
  /**
   * Setter which also initialize computing of crowding distance.
   * @param pop population
   */
  public void setPopulation(final EvPopulation<T> pop) {
    this.population = pop;
    this.crowdingDistance = new double[population.size()];
    computeCrowdingDistance();
  }
  
  /**
   * Computes crowding distance for each individual in population
   * and remembers it in table in object of this class.
   */
  private void computeCrowdingDistance() {
    Element[] elements = new Element[population.size()];
    int population_size = population.size();
    for (int i = 0; i < population_size; i++) {
      elements[i] = new Element(i);
    }
    for (int i = 0; i < population.get(0).getObjectiveFunctions().size(); i++) {
      for (int j = 0; j < population.size(); j++) {
        elements[j].value = population.
        get(elements[j].index).getObjectiveFunctionValue(i);
      }
      Arrays.sort(elements);
      elements[0].distance = Double.MAX_VALUE;
      elements[elements.length - 1].distance = Double.MAX_VALUE;
      for (int j = 1; j < population.size() - 1; j++) {
        if (elements[j].distance != Double.MAX_VALUE) {
          elements[j].distance += elements[j + 1].value - elements[j - 1].value;
        }
      }
    }
    for (int i = 0; i < population_size; i++) {
      crowdingDistance[elements[i].index] = elements[i].distance;
    }
  }
}
