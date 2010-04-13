package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;


/**
 * Solution Space for MIMIC.
 * 
 * EXPERIMENTAL - USE AT OWN RISK !!
 * Based on EvBinaryVectorMIMICSpace by Sabina Fabiszewska
 * 
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 */
public class EvDiscreteVectorMIMICSpace implements
    EvSolutionSpace<EvKnaryIndividual> {


  /**
   * {@inheritDoc}
   */
  private static final long serialVersionUID = 7873093419696652852L;

  /**
   * Network of relation between variables.
   */
  private final EvDiscreteVectorMIMICBayesianNetwork network;

  /**
   * Objective function.
   */
  private EvObjectiveFunction<EvKnaryIndividual> function;


  /**
   * Constructor.
   * 
   * @param bayesian_network bayesian network
   */
  public EvDiscreteVectorMIMICSpace(
      final EvDiscreteVectorMIMICBayesianNetwork bayesian_network) {
    network = bayesian_network;
  }


  /**
   * {@inheritDoc}
   */
  public EvKnaryIndividual generateIndividual() {
    return network.generateIndividual();
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvKnaryIndividual> getObjectiveFuntion() {
    return function;
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
    return null;
  }

  
}
