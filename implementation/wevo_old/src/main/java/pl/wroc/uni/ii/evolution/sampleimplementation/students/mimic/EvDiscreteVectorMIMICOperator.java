package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;


/**
 * MIMIC Operator.
 * 
 * EXPERIMENTAL - USE AT OWN RISK !!
 * 
 * @version 0.1
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 */
public class EvDiscreteVectorMIMICOperator implements 
    EvOperator<EvKnaryIndividual> {

  /**
   * Individual dimension.
   */
  private final int dimension;

  /**
   * Size of a population.
   */
  private final int population_size;

  /**
   * Table of possible gene values.
   */
  private final int[] geneValues;
  
  /**
   * Network of relation between variables.
   */
  private final EvDiscreteVectorMIMICBayesianNetwork network;

  /**
   * Solution Space.
   */
  private final EvDiscreteVectorMIMICSpace space;


  /**
   * Constructor.
   * 
   * @param dim individual dimension
   * @param gVal table of possible gene values
   * @param pop_size size of population which should be generated
   * @param bayesian_network
   */
  public EvDiscreteVectorMIMICOperator(final int dim, final int[] gVal, 
      final int pop_size) {

    dimension = dim;
    population_size = pop_size;
    geneValues = gVal;
    network = new EvDiscreteVectorMIMICBayesianNetwork(dimension, geneValues);

    space = new EvDiscreteVectorMIMICSpace(network);
  }


  /**
   * @return Solution Space for MIMIC
   */
  public EvDiscreteVectorMIMICSpace getSolutionSpace() {
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
