package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Multipoint recombination for natureal number individuals
 * 
 * @author Kamil Dworakowski, Marcin Golebiowski
 */
public class EvNaturalNumberVectorMultiPointCrossover extends
    EvCrossover<EvNaturalNumberVectorIndividual> {

  private int points = 2;


  public EvNaturalNumberVectorMultiPointCrossover() {
  }


  public EvNaturalNumberVectorMultiPointCrossover(int num_of_points) {
    points = num_of_points;
  }


  public int arity() {
    return 2;
  }


  public List<EvNaturalNumberVectorIndividual> combine(
      List<EvNaturalNumberVectorIndividual> individuals) {

    assert individuals.size() == 2;

    List<EvNaturalNumberVectorIndividual> result =
        new ArrayList<EvNaturalNumberVectorIndividual>();

    int dimension = individuals.get(0).getDimension();

    if (points > dimension) {
      throw new RuntimeException("Too many recombination points");
    }

    boolean[] swap = EvRandomizer.INSTANCE.nextBooleanList(dimension, points);
    EvNaturalNumberVectorIndividual indiv1, indiv2, swapper;
    EvNaturalNumberVectorIndividual baby1 =
        new EvNaturalNumberVectorIndividual(dimension);
    EvNaturalNumberVectorIndividual baby2 =
        new EvNaturalNumberVectorIndividual(dimension);
    indiv1 = individuals.get(0);
    indiv2 = individuals.get(1);
    for (int i = 0; i < dimension; i++) {
      if (swap[i]) {
        swapper = indiv2;
        indiv2 = indiv1;
        indiv1 = swapper;
      }
      baby1.setNumberAtPosition(i, indiv1.getNumberAtPosition(i));
      baby2.setNumberAtPosition(i, indiv2.getNumberAtPosition(i));

    }
    baby1.setObjectiveFunction(indiv1.getObjectiveFunction());
    baby2.setObjectiveFunction(indiv1.getObjectiveFunction());
    result.add(baby1);
    result.add(baby2);
    return result;
  }


  @Override
  public int combineResultSize() {
    return 2;
  }

}
