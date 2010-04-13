package pl.wroc.uni.ii.evolution.engine.operators.general.selections.multiobjective;

import java.util.Arrays;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Class use to compute rank values for each individual in population.
 * If individual is dominated by all other individuals then his rank is 1
 * and when it is dominated by lower number of individuals he gets bigger rank.
 * Individual is dominated if each of his objective functions value is smaller
 * then value of each the same objective function of compared individual.
 * 
 * @author Adam Palka
 * @param <T> - type of individuals
 */
public class EvParetoFrontsRank<T extends EvIndividual> {
  
  /** Population for which we want to compute rank values. */
  private EvPopulation<T> population;
  /** Table witch rank of each individual. */
  private int[] rank;

  /** Class used to remember indexes of individuals and sorting. */
  private static class Element implements Comparable<Element> {
    
    /** Individual number. */
    private int index;
    /** Value of objective function used for sorting. */
    private double value;
    /** Pareto front number. */
    private int paretoFront;
    
    /**
     * Constructor.
     * @param i -individual number
     * @param v -objective function value
     */
    public Element(final int i, final double v) {
      this.index = i;
      this.value = v;
      this.paretoFront = 0;
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
   * Constructor which also initialize computing of ranks.
   * @param pop -population 
   */
  public EvParetoFrontsRank(final EvPopulation<T> pop) {
    this.population = pop;
    this.rank = new int[population.size()];
    computeRank();
  }

  /**
   * Getter.
   * @param k -number of individual.
   * @return rank value of individual k.
   */
  public int getRank(final int k) {
    return rank[k];
  }
  
  /**
   * Setter which also initialize computing of ranks.
   * @param pop population
   */
  public void setPopulation(final EvPopulation<T> pop) {
    this.population = pop;
    this.rank = new int[population.size()];
    computeRank();
  }
  
  /**
   * Computes rank for each individual in population.
   */
  private void computeRank() {
    
    int population_size = population.size();
    Element[] elements = new Element[population_size];
    for (int i = 0; i < population_size; i++) {
      elements[i] = new Element(i, population.
          get(i).getObjectiveFunctionValue(0));
    }
    Arrays.sort(elements);
    for (int i = population_size - 2; i >= 0; i--) {
      for (int j = i + 1; j < population_size; j++) {
        if (population.get(elements[i].index).
            compareTo(population.get(elements[j].index)) == -1) {
          elements[i].paretoFront++;
        }
      }
    }
    /** MAX Pareto front value in population. */
    int maxPF = elements[0].paretoFront;
    for (int i = 1; i < population_size; i++) {
      if (elements[i].paretoFront > maxPF) {
        maxPF = elements[i].paretoFront;
      }
    }
    for (int i = 0; i < population_size; i++) {
      rank[elements[i].index] = maxPF - elements[i].paretoFront + 1;
    }
  }

}
