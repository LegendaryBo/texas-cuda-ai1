package pl.wroc.uni.ii.evolution.sampleimplementation.students.sabinafabiszewska;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;

/**
 * @author Sabina Fabiszewska
 */
public class EvMyAlgorithm extends EvAlgorithm<EvMyIndividual> {

  /**
   * 
   */
  private final int dimension;

  /**
   * 
   */
  private final int max_iteration;

  /**
   * 
   */
  private final double mutation_probability;


  /**
   * @param dim dimension of chromosome
   * @param pop_size size of population
   * @param max_it number of iterations
   * @param mut_probability probability of mutation
   */
  public EvMyAlgorithm(final int dim, final int pop_size, final int max_it,
      final double mut_probability) {
    super(pop_size);
    this.dimension = dim;
    this.population_size = pop_size;
    this.max_iteration = max_it;
    this.mutation_probability = mut_probability;
  }


  /**
   * 
   */
  @Override
  public void init() {
    setSolutionSpace(new EvMySpace(dimension));
    setObjectiveFunction(new EvMyObjectiveFunction());
    setTerminationCondition(new EvMaxIteration<EvMyIndividual>(max_iteration));
    addOperatorToEnd(new EvMyCrossover());
    addOperatorToEnd(new EvMyMutation(mutation_probability));
    super.init();
  }

}
