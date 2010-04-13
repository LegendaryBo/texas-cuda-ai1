package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Uniform crossover for EvBinaryIndividuals. In each call of combine() the
 * operator selects 2 individuals from population and create 2 new child
 * individuals in following way:<br>
 * For each gene in child shuffle boolean. If it's copy gene from parent1,
 * otherwise copy gene from parent2.<br>
 * The operator calls combine() as many times as it's neccesary match number of
 * individuals defined by EvCrossoverApplyStrategy (by default
 * EvPersistPopulationSizeApplyStrategy).<br>
 * Parents are selected using EvCombineParentSelector (default is
 * EvSimpleCombineSelector), to change it use setCombineParentSelector()
 * function<br>
 * <br>
 * Example: (results are random)<br>
 * Parent 1: (6,6,6,6,6) Parent 2: (2,2,2,2,2)<br>
 * Child 1: (6,6,2,6,2) Child 2: (2,2,6,2,6)<br>
 * 
 * @author Marcin Golebiowski
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvKnaryVectorUniformCrossover<T extends EvKnaryIndividual> extends
    EvCrossover<T> {

  @Override
  public int arity() {
    return 2;
  }


  @Override
  public List<T> combine(List<T> parents) {
    assert parents.size() == 2;

    List<T> result = new ArrayList<T>(2);
    /**
     * Two parents
     */
    T parent1 = parents.get(0);
    T parent2 = parents.get(1);

    int dimension = parent1.getDimension();

    // create child individual by cloning input individuals
    T baby1 = (T) parent1.clone();
    T baby2 = (T) parent2.clone();

    // clone doesn't copy objective function
    baby1.setObjectiveFunction(parent1.getObjectiveFunction());
    baby2.setObjectiveFunction(parent2.getObjectiveFunction());

    // for each gene in child individuals shuffle boolean to decide from
    // which parent it will be copied
    for (int position = 0; position < dimension; position++) {
      int master_parent = EvRandomizer.INSTANCE.nextBoolean() ? 0 : 1;

      baby1.setGene(position, (master_parent == 0) ? parent1.getGene(position)
          : parent2.getGene(position));
      baby2.setGene(position, (master_parent == 0) ? parent2.getGene(position)
          : parent1.getGene(position));
    }

    result.add(baby1);
    result.add(baby2);

    return result;
  }


  @Override
  public int combineResultSize() {
    return 2;
  }

}
