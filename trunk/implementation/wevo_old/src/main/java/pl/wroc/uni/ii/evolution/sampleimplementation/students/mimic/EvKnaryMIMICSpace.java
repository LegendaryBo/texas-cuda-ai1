package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;

/**
 * Solution Space for MIMIC - k-nary version.
 * 
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 *
 */
public class EvKnaryMIMICSpace implements EvSolutionSpace<EvKnaryIndividual> {

  
  /**
   * {@inheritDoc}
   */
  private static final long serialVersionUID = -6442466889726073202L;

  /**
   * Network of relation between variables.
   */
  private final EvKnaryMIMICBayesianNetwork network;

  /**
   * Objective function.
   */
  private EvObjectiveFunction<EvKnaryIndividual> function;


  /**
   * Constructor.
   * 
   * @param bayesian_network bayesian network
   */
  public EvKnaryMIMICSpace(
      final EvKnaryMIMICBayesianNetwork bayesian_network) {
    network = bayesian_network;
  }


  /**
   * {@inheritDoc}
   */
  public EvKnaryIndividual generateIndividual() {
    EvKnaryIndividual individual = network.generateIndividual();
    individual.setObjectiveFunction(function);
    return individual;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvKnaryIndividual> getObjectiveFuntion() {
    return this.getObjectiveFuntion();
  }


  /**
   * {@inheritDoc}
   */
  
  public void setObjectiveFuntion(
      final EvObjectiveFunction<EvKnaryIndividual> objective_function) {
    function = objective_function;
  }


  /**
   * not implemented. {@inheritDoc}
   */
  public boolean belongsTo(final EvKnaryIndividual individual) {
    return false;
  }


  /**
   * not implemented. {@inheritDoc}
   */
  public Set<EvSolutionSpace<EvKnaryIndividual>> divide(final int n) {
    return null;
  }


  /**
   * not implemented. {@inheritDoc}
   */
  public Set<EvSolutionSpace<EvKnaryIndividual>> divide(final int n,
      final Set<EvKnaryIndividual> p) {
    return null;
  }


  /**
   * not implemented. {@inheritDoc}
   */
  public EvKnaryIndividual takeBackTo(
      final EvKnaryIndividual individual) {
    return generateIndividual();
  }
}
