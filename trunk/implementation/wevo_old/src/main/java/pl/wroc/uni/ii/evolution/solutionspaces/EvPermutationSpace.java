package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.ArrayList;
import java.util.Set;
import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A solution space of EvPermutationIndividuals.
 * 
 * @author Donata Malecka
 * @author Piotr Baraniak
 */
public class EvPermutationSpace implements
    EvSolutionSpace<EvPermutationIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = -3540767034804975908L;

  /**
   * Permutation length, number of genes.
   */
  private int dimension;

  /**
   * Objective function used which will be applied to individuals generated 
   * by  this solution space.
   */
  private EvObjectiveFunction<EvPermutationIndividual> objective_function;

  /**
   * 
   * 
   * @param size - length of permutation.
   */
  public EvPermutationSpace(int size) {
    dimension = size;
  }


  public Set<EvSolutionSpace<EvPermutationIndividual>> divide(int n) {
    // TODO implement
    return null;
  }


  public Set<EvSolutionSpace<EvPermutationIndividual>> divide(int n, Set p) {
    // TODO implement
    return null;
  }


  public boolean belongsTo(EvPermutationIndividual individual) {
    // TODO implement
    return false;
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvPermutationIndividual> objective_function) {
    this.objective_function = objective_function;

  }


  public EvPermutationIndividual takeBackTo(EvPermutationIndividual individual) {
    // TODO Auto-generated method stub
    return null;
  }


  public EvPermutationIndividual generateIndividual() {
    int[] chromosome = new int[dimension];
    ArrayList<Integer> to_random = new ArrayList<Integer>();
    for (int i = 0; i < dimension; i++) {
      to_random.add(i);
    }
    for (int i = 0; i < dimension; i++) {
      int j = EvRandomizer.INSTANCE.nextInt(to_random.size());
      chromosome[i] = (int) to_random.get(j);
      to_random.remove(j);
    }
    EvPermutationIndividual ind = new EvPermutationIndividual(chromosome);
    ind.setObjectiveFunction(objective_function);
    return ind;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvPermutationIndividual> getObjectiveFuntion() {
    return objective_function;
  }

}
