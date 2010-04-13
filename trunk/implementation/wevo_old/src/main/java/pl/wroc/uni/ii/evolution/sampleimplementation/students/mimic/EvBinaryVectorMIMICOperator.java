package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * MIMIC Operator - binary version.
 * 
 * @version 0.01 beta
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 */
public class EvBinaryVectorMIMICOperator implements
    EvOperator<EvBinaryVectorIndividual> {

  
  /**
   * Individual dimension.
   */
  private final int dimension;

  /**
   * Size of a population.
   */
  private final int population_size;

  /**
   * Network of relation between variables.
   */
  private final EvBinaryVectorMIMICBayesianNetwork network;

  /**
   * Solution Space.
   */
  private final EVBinaryVectorMIMICSpace space;
  
  /**
   * Objective function.
   */
  private final 
  EvObjectiveFunction<EvBinaryVectorIndividual> objective_function;


  /**
   * Constructor.
   * 
   * @param dim individual dimension
   * @param pop_size size of population which should be generated
   * @param bayesian_network
   * @param function objective function
   */
  public EvBinaryVectorMIMICOperator(final int dim, final int pop_size,
      final EvObjectiveFunction<EvBinaryVectorIndividual> function) {
    
    dimension = dim;
    population_size = pop_size;
    objective_function = function; 
    
    network = new EvBinaryVectorMIMICBayesianNetwork(dimension, 
        objective_function);
    space = new EVBinaryVectorMIMICSpace(network);
  }


  /**
   * @return Solution Space for MIMIC
   */
  public EVBinaryVectorMIMICSpace getSolutionSpace() {
    return space;
  }


  /**
   * {@inheritDoc}
   */
  public EvPopulation<EvBinaryVectorIndividual> apply(
      final EvPopulation<EvBinaryVectorIndividual> population) {
    
    network.estimateProbabilities(population);

    EvPopulation<EvBinaryVectorIndividual> new_population =
        new EvPopulation<EvBinaryVectorIndividual>();
    for (int i = 0; i < population_size; i++) {
      new_population.add(network.generateIndividual());
    }

    return new_population;
  }

}
