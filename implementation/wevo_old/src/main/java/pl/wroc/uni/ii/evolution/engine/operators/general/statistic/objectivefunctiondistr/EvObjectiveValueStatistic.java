package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr;

import java.io.Serializable;

/**
 * A statistic about fitness value in some population. It contains number of
 * individuals having this fitness value.
 * 
 * @author Marcin Golebiowski
 */
public class EvObjectiveValueStatistic implements Serializable {

  /**
   * 
   */
  private static final long serialVersionUID = 3273292324325429494L;

  private double fitness;

  private int number;


  public EvObjectiveValueStatistic(double fitness, int number) {
    this.fitness = fitness;
    this.number = number;
  }


  public double getFitness() {
    return fitness;

  }


  public int getNumber() {
    return number;
  }

}
