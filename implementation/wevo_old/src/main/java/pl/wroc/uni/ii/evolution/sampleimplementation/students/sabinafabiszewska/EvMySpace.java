package pl.wroc.uni.ii.evolution.sampleimplementation.students.sabinafabiszewska;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;

/**
 * @author Sabina Fabiszewska
 */
public class EvMySpace implements EvSolutionSpace<EvMyIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = 4018741190338778171L;

  /**
   * 
   */
  private final int dimension;

  /**
   * 
   */
  private EvObjectiveFunction<EvMyIndividual> objective_function;


  /**
   * @param dim dimension of solution space
   */
  public EvMySpace(final int dim) {
    this(new EvMyObjectiveFunction(), dim);
  }


  /**
   * @param object_function objective funkction
   * @param dim dimension
   */
  public EvMySpace(final EvObjectiveFunction<EvMyIndividual> object_function,
      final int dim) {
    this.dimension = dim;
    setObjectiveFuntion(object_function);
  }


  /**
   * @param individual individual
   * @return if individual belongs to this solution space
   */
  public boolean belongsTo(final EvMyIndividual individual) {
    return true;
  }


  /**
   * [not used in current version].
   * 
   * @param n -
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvMyIndividual>> divide(final int n) {
    return null;
  }


  /**
   * [not used in current version].
   * 
   * @param n -
   * @param p -
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvMyIndividual>> divide(final int n,
      final Set<EvMyIndividual> p) {
    return null;
  }


  /**
   * @return new generated individual
   */
  public EvMyIndividual generateIndividual() {
    return new EvMyIndividual(dimension);
  }


  /**
   * @return objective function used in this solution space
   */
  public EvObjectiveFunction<EvMyIndividual> getObjectiveFuntion() {
    return objective_function;
  }


  /**
   * @param object_function objective function
   */
  public void setObjectiveFuntion(
      final EvObjectiveFunction<EvMyIndividual> object_function) {
    this.objective_function = objective_function;
  }


  /**
   * @param individual bad individual
   * @return new individual which belongs to solution space
   */
  public EvMyIndividual takeBackTo(final EvMyIndividual individual) {
    return individual;
  }
}
