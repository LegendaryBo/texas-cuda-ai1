package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;

/**
 * This n-ary operator's combine takes a population of size at most n, and
 * returns an avarage individual. <br>
 * <b> WARNING: </b> Result of apply with default EvCrossoverApplyStrategy
 * (EvPersistSizeApplyStrategy) is population with same individuals.
 * <p>
 * Default arity is 2.
 * 
 * @author Kamil Dworakowski, Jarek Fuks
 */
public class EvNaturalNumberVectorAverageCrossover extends
    EvCrossover<EvNaturalNumberVectorIndividual> {
  private int arity;


  /**
   * Constructor.
   * 
   * @param arity how many individuals are needed by this operator to create a
   *        child
   */
  public EvNaturalNumberVectorAverageCrossover(int arity) {
    this.arity = arity;
  }


  /**
   * Default arity is 2.
   */
  public EvNaturalNumberVectorAverageCrossover() {
    arity = 2;
  }


  @Override
  public int arity() {
    return arity;
  }


  @Override
  public List<EvNaturalNumberVectorIndividual> combine(
      List<EvNaturalNumberVectorIndividual> parents) {

    List<EvNaturalNumberVectorIndividual> result =
        new ArrayList<EvNaturalNumberVectorIndividual>();

    EvNaturalNumberVectorIndividual baby =
        new EvNaturalNumberVectorIndividual(parents.get(0).getDimension());

    // baby = x1 + x2 +...+xn
    for (int i = 0; i < arity; i++) {
      for (int dim = 0; dim < baby.getDimension(); dim++) {

        int avarage_gene =
            baby.getNumberAtPosition(dim)
                + parents.get(i).getNumberAtPosition(dim);

        baby.setNumberAtPosition(dim, avarage_gene);
      }
    }

    // baby[i] <- baby[i] / population.size
    for (int dim = 0; dim < baby.getDimension(); dim++) {
      baby.setNumberAtPosition(dim, baby.getNumberAtPosition(dim)
          / parents.size());
    }

    baby.setObjectiveFunction(parents.get(0).getObjectiveFunction());

    result.add(baby);

    return result;

  }


  @Override
  public int combineResultSize() {
    return 1;
  }

}
