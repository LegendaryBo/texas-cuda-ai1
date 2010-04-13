package pl.wroc.uni.ii.evolution.sampleimplementation.students.sabinafabiszewska;

import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Sabina Fabiszewska
 */
public class EvMyCrossover extends EvCrossover<EvMyIndividual> {

  /**
   * 
   */
  private static final int ARITY = 2;


  /**
   * @return operator arity
   */
  @Override
  public int arity() {
    return ARITY;
  }


  /**
   * @param parents list of 2 parents used to generate 2 children
   * @return list of 2 children generated by operator EvMyCrossover
   */
  @Override
  public List<EvMyIndividual> combine(final List<EvMyIndividual> parents) {

    EvMyIndividual parent1 = parents.get(0);
    EvMyIndividual parent2 = parents.get(1);

    int dimension = parent1.getDimension();
    int point = EvRandomizer.INSTANCE.nextInt(0, dimension);

    EvMyIndividual child1 = new EvMyIndividual(dimension);
    EvMyIndividual child2 = new EvMyIndividual(dimension);

    for (int i = 0; i < dimension; i++) {
      if (i <= point) {
        child1.setBit(i, parent1.getBit(i));
        child2.setBit(i, parent2.getBit(i));
      } else {
        child1.setBit(i, parent2.getBit(i));
        child2.setBit(i, parent1.getBit(i));
      }
    }

    List<EvMyIndividual> children = new ArrayList<EvMyIndividual>(2);
    children.add(child1);
    children.add(child2);

    return children;
  }


  /**
   * @return length of list returned by method 'combine'
   */
  @Override
  public int combineResultSize() {
    return ARITY;
  }

}