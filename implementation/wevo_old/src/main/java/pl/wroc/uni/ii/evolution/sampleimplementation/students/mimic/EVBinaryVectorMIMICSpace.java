package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;

/**
 * Solution Space for MIMIC - binary version.
 * 
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 */
public class EVBinaryVectorMIMICSpace implements
    EvSolutionSpace<EvBinaryVectorIndividual> {

  
  /**
   * {@inheritDoc}
   */
  private static final long serialVersionUID = 7873093419696652852L;

  /**
   * Network of relation between variables.
   */
  private final EvBinaryVectorMIMICBayesianNetwork network;

  /**
   * Objective function.
   */
  private EvObjectiveFunction<EvBinaryVectorIndividual> function;


  /**
   * Constructor.
   * 
   * @param bayesian_network bayesian network
   */
  public EVBinaryVectorMIMICSpace(
      final EvBinaryVectorMIMICBayesianNetwork bayesian_network) {
    network = bayesian_network;
  }


  /**
   * {@inheritDoc}
   */
  public EvBinaryVectorIndividual generateIndividual() {
    EvBinaryVectorIndividual individual = network.generateIndividual();
    individual.setObjectiveFunction(function);
    return individual;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvBinaryVectorIndividual> getObjectiveFuntion() {
    return this.getObjectiveFuntion();
  }


  /**
   * {@inheritDoc}
   */
  
  public void setObjectiveFuntion(
      final EvObjectiveFunction<EvBinaryVectorIndividual> objective_function) {
    function = objective_function;
  }


  /**
   * not implemented. {@inheritDoc}
   */
  public boolean belongsTo(final EvBinaryVectorIndividual individual) {
    return false;
  }


  /**
   * not implemented. {@inheritDoc}
   */
  public Set<EvSolutionSpace<EvBinaryVectorIndividual>> divide(final int n) {
    return null;
  }


  /**
   * not implemented. {@inheritDoc}
   */
  public Set<EvSolutionSpace<EvBinaryVectorIndividual>> divide(final int n,
      final Set<EvBinaryVectorIndividual> p) {
    return null;
  }


  /**
   * not implemented. {@inheritDoc}
   */
  public EvBinaryVectorIndividual takeBackTo(
      final EvBinaryVectorIndividual individual) {
    return generateIndividual();
  }
}
