package pl.wroc.uni.ii.evolution.solutionspaces;

import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import java.util.Set;

/**
 * A solution space for EvOmegaIndividual
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaSpace implements EvSolutionSpace<EvOmegaIndividual> {

  private static final long serialVersionUID = -3549759084076703480L;

  // size of the problem -- it determins genotype length
  private int problem_size;

  // objective function
  private EvObjectiveFunction<EvOmegaIndividual> objective_function;


  /**
   * Constructor It creates a solution space with given problem size and
   * objective function
   * 
   * @param problem_size size of the problem
   * @param objective_function objective function
   */
  public EvOmegaSpace(int problem_size,
      EvObjectiveFunction<EvOmegaIndividual> objective_function) {
    this.problem_size = problem_size;
    this.objective_function = objective_function;
  }


  public Set<EvSolutionSpace<EvOmegaIndividual>> divide(int n) {
    // TODO implement
    return null;
  }


  public Set<EvSolutionSpace<EvOmegaIndividual>> divide(int n, Set p) {
    // TODO implement
    return null;
  }


  public boolean belongsTo(EvOmegaIndividual individual) {
    // TODO implement
    return false;
  }


  public void setObjectiveFuntion(
      EvObjectiveFunction<EvOmegaIndividual> objective_function) {
    this.objective_function = objective_function;
  }


  public EvObjectiveFunction<EvOmegaIndividual> getObjectiveFuntion() {
    return objective_function;
  }


  public EvOmegaIndividual takeBackTo(EvOmegaIndividual individual) {
    // TODO implement
    return null;
  }


  public EvOmegaIndividual generateIndividual() {
    return new EvOmegaIndividual(problem_size, null, objective_function);
  }
}
