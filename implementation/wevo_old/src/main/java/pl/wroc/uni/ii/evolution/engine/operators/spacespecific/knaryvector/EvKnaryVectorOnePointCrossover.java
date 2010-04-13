package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Crossover for EvKnaryIndividuals, which uses 2 parents to create 2 children
 * with one point cut.<br>
 * Each call of combine() constructs 2 new individuals, The operator calls
 * combine() as many times as it's neccesary match number of individuals defined
 * by EvCrossoverApplyStrategy (by default
 * EvPersistPopulationSizeApplyStrategy). <br>
 * Parents are selected using EvCombineParentSelector (default is
 * EvSimpleCombineSelector), to change it use setCombineParentSelector()
 * function<br>
 * <br>
 * Example:<br>
 * Consider we have individuals:<br>
 * 3,4,3,4,3,4 and 9,9,9,9,9,9<br>
 * Crossover shuffle break point and produces two NEW individuals, like:<br>
 * 3,4,3,4,9,9 and 9,9,9,9,3,4
 * 
 * @author Piotr Baraniak
 * @author Marek Chrusciel
 * @author Marcin Golebiowski
 * @author Kacper Gorski (admin@34all.org)
 */

public class EvKnaryVectorOnePointCrossover<T extends EvKnaryIndividual>
    extends EvCrossover<T> {

  @Override
  public int arity() {
    return 2;
  }


  @Override
  public List<T> combine(List<T> args) {
    assert args.size() == 2;

    T old_ind1 = args.get(0);
    T old_ind2 = args.get(1);

    int dimension = old_ind1.getDimension();

    // result list
    List<T> children = new ArrayList<T>(2);

    // new individuals are cloned from old ones
    // It's done in strange way because we can't call constructor
    // of type T
    T ind1 = (T) old_ind1.clone();
    T ind2 = (T) old_ind2.clone();

    children.add(ind1);
    children.add(ind2);

    // shuffling break point
    int break_point = EvRandomizer.INSTANCE.nextInt(dimension - 1);

    for (int i = 0; i <= break_point; i++) {
      ind1.setGene(i, old_ind1.getGene(i));
      ind2.setGene(i, old_ind2.getGene(i));
    }

    for (int i = break_point + 1; i < dimension; i++) {
      ind2.setGene(i, old_ind1.getGene(i));
      ind1.setGene(i, old_ind2.getGene(i));
    }

    // need to set objective function since clone() don't do that
    ind1.setObjectiveFunction(old_ind1.getObjectiveFunction());
    ind2.setObjectiveFunction(old_ind1.getObjectiveFunction());

    return children;
  }


  @Override
  public int combineResultSize() {
    return 2;
  }

}
