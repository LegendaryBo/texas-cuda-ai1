package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Uniform crossover for EvNaturalNumberVectorIndividuals.
 * 
 * @author Marcin Golebiowski
 */
public class EvNaturalNumberVectorUniformCrossover extends
    EvCrossover<EvNaturalNumberVectorIndividual> {

  @Override
  public List<EvNaturalNumberVectorIndividual> combine(
      List<EvNaturalNumberVectorIndividual> parents) {

    List<EvNaturalNumberVectorIndividual> result =
        new ArrayList<EvNaturalNumberVectorIndividual>();

    /** Two parents */
    EvNaturalNumberVectorIndividual parent1 = parents.get(0);
    EvNaturalNumberVectorIndividual parent2 = parents.get(1);

    /** Children * */
    EvNaturalNumberVectorIndividual child1 =
        new EvNaturalNumberVectorIndividual(parent1.getDimension());
    child1.setObjectiveFunction(parent1.getObjectiveFunction());

    EvNaturalNumberVectorIndividual child2 =
        new EvNaturalNumberVectorIndividual(parent1.getDimension());
    child2.setObjectiveFunction(parent1.getObjectiveFunction());

    for (int position = 0; position < parent1.getDimension(); position++) {
      int master_parent = EvRandomizer.INSTANCE.nextBoolean() ? 0 : 1;

      child1.setNumberAtPosition(position, (master_parent == 0) ? parent1
          .getNumberAtPosition(position) : parent2
          .getNumberAtPosition(position));
      child2.setNumberAtPosition(position, (master_parent == 0) ? parent2
          .getNumberAtPosition(position) : parent1
          .getNumberAtPosition(position));
    }

    result.add(child1);
    result.add(child2);

    return result;
  }


  @Override
  public int arity() {
    return 2;
  }


  @Override
  public int combineResultSize() {
    return 2;
  }

}
