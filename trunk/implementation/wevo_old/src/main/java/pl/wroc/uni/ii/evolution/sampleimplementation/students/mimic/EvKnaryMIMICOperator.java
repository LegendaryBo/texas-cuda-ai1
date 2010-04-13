package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * MIMIC Operator used in MIMIC algorithm - k-nary version.
 * 
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 */
public class EvKnaryMIMICOperator implements EvOperator<EvKnaryIndividual> {

  
  /**
   * Individual dimension.
   */
  private final int dimension;

  /**
   * Size of a population.
   */
  private final int population_size;
  
  /**
   * Number of values which variable can assume.
   * Set of values: {0, 1, ... numberOfValues-1}
   */
  private final int numberOfValues;

  /**
   * Network of relation between variables.
   */
  private final EvKnaryMIMICBayesianNetwork network;

  /**
   * Solution Space.
   */
  private final EvKnaryMIMICSpace space;
  
  /**
   * Objective function.
   */
  private final 
  EvObjectiveFunction<EvKnaryIndividual> objective_function;


  /**
   * Constructor.
   * 
   * @param dim individual dimension
   * @param pop_size size of population which should be generated
   * @param values number of possible values
   * @param function objective function
   */
  public EvKnaryMIMICOperator(final int dim, final int pop_size, 
      final int values,
      final EvObjectiveFunction<EvKnaryIndividual> function) {
    
    dimension = dim;
    population_size = pop_size;
    objective_function = function;
    numberOfValues = values;
    
    network = new EvKnaryMIMICBayesianNetwork(dimension, numberOfValues, 
        objective_function);
    space = new EvKnaryMIMICSpace(network);
  }


  /**
   * @return Solution Space for MIMIC
   */
  public EvKnaryMIMICSpace getSolutionSpace() {
    return space;
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<EvKnaryIndividual> apply(
      final EvPopulation<EvKnaryIndividual> population) {
    
    network.estimateProbabilities(population);

    EvPopulation<EvKnaryIndividual> new_population =
        new EvPopulation<EvKnaryIndividual>();
    for (int i = 0; i < population_size; i++) {
      new_population.add(network.generateIndividual());
    }

    return new_population;
  }
}
