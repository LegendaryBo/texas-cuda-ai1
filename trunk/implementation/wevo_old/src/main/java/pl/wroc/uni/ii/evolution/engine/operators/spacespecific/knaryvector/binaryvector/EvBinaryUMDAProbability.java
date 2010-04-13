/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Operator for umda computing probability vector.
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public class EvBinaryUMDAProbability implements
    EvOperator<EvBinaryVectorIndividual> {

  /** Size of new population. */
  private final int populationSize;


  /**
   * Constructor.
   * 
   * @param population_size_ Size of new population
   */
  public EvBinaryUMDAProbability(final int population_size_) {
    this.populationSize = population_size_;
  }


  /**
   * Generate new population. 1. Compute for each gene probability to get 1
   * depend on the population. 2. Generate randomly new population. The size of
   * new population has been set in the constructor. Each gene of each 
   * individual is randomly generate depends on vector of probability.
   * 
   * @param population The best individuals from population.
   * @return New population - size = this.population_size
   */
  public EvPopulation<EvBinaryVectorIndividual> apply(
      final EvPopulation<EvBinaryVectorIndividual> population) {
    double[] probability = null;
    if (population == null || population.size() <= 0) {
      throw new IllegalArgumentException(
          "Population size must be grater than 0.");
    }

    EvBinaryVectorSpace space =
        new EvBinaryVectorSpace(null, population.get(0).getDimension());
    EvObjectiveFunction objective_function =
        population.get(0).getObjectiveFunction();
    // compute probability vector:
    probability = computeProbability(population);

    EvPopulation<EvBinaryVectorIndividual> newPopulation =
        new EvPopulation<EvBinaryVectorIndividual>();

    // Generating individuals
    for (int i = 0; i < this.populationSize; i++) {
      EvBinaryVectorIndividual in =
          (EvBinaryVectorIndividual) space.generateIndividual(probability);
      in.setObjectiveFunction(objective_function);
      newPopulation.add(in);
    }

    return newPopulation;
  }


  /**
   * Compute probability to get 1 for each gene in chromosome. e.g. Three
   * individuals at second gene have 1 Seven individuals at second gene have 0 
   * The probability for second gene is: 0.3
   * 
   * @param population Population.
   * @return Probability vector for genes.
   */
  public double[] computeProbability(
      final EvPopulation<EvBinaryVectorIndividual> population) {

    double[] probability = new double[population.get(0).getDimension()];

    for (int i = 0; i < probability.length; i++) {
      int counted = 0; // counted genes with 1
      for (int j = 0; j < population.size(); j++) {
        if (population.get(j).getGene(i) == 1) {
          counted++;
        }
      }
      probability[i] = (double) counted / (double) population.size();
    }
    return probability;
  }

}
