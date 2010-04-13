package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Multi-point crossover for EvKnaryIndividuals.<br>
 * The operator works in similar way like EvKnaryOnePointCrossover operator. It
 * breaks 2 parents in specified number of parts and then creates children by
 * merging parts from 2 different parent.<br>
 * <br>
 * Example:<br>
 * Consider we have 2 parents and we cut them in 3 places.<br>
 * Parent 1: (0,0,0,0,0,0,0,0,0,0) Parent: 2 (2,2,2,2,2,2,2,2,2,2)<br>
 * operator shuffled break point at indexes 1, 3 and 6<br>
 * Child 1: (0,2,2,0,0,0,2,2,2,2) Child 1: (2,0,0,2,2,2,0,0,0,0)<br>
 * 
 * @author Jarek Fuks (jarek102@gmail.com)
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @author Kacper Gorski (admin@34all.org)
 * @param <T> - type of individuals the operator works on
 */

public class EvKnaryVectorMultiPointCrossover<T extends EvKnaryIndividual>
    extends EvCrossover<T> {

  private int points = 2;


  /**
   * Constructor the operator which will create individuals from parents by
   * breaking them exacly in <b>num_of_points</b> places.
   * 
   * @param num_of_points number of cut points
   */
  public EvKnaryVectorMultiPointCrossover(int num_of_points) {

    if (num_of_points <= 0) {
      throw new IllegalArgumentException(
          "Number of cut points must be greater than 0");
    }
    points = num_of_points;
  }


  @Override
  public int arity() {
    return 2;
  }


  @Override
  public List<T> combine(List<T> parents) {
    assert parents.size() == 2;

    List<T> result = new ArrayList<T>(2);

    int dimension = parents.get(0).getDimension();

    if (points > dimension)
      throw new RuntimeException("Too many recombination points");

    boolean[] swap = EvRandomizer.INSTANCE.nextBooleanList(dimension, points);

    T parent1 = parents.get(0);
    T parent2 = parents.get(1);

    // creating new individuals by cloning because we can't call T constructor
    T baby1 = (T) parent1.clone();
    T baby2 = (T) parent2.clone();

    // temporary reference for swapping
    T swapper = null;

    for (int i = 0; i < dimension; i++) {
      if (swap[i]) {
        swapper = parent2;
        parent2 = parent1;
        parent1 = swapper;
      }
      baby1.setGene(i, parent1.getGene(i));
      baby2.setGene(i, parent2.getGene(i));
    }

    baby1.setObjectiveFunction(parent1.getObjectiveFunction());
    baby2.setObjectiveFunction(parent1.getObjectiveFunction());

    result.add(baby1);
    result.add(baby2);

    return result;

  }


  @Override
  public int combineResultSize() {
    return 2;
  }

}
