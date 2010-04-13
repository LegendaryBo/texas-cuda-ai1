package pl.wroc.uni.ii.evolution.solutionspaces;

import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Solution space containing specified number of genes, each of them is an
 * integer, whose value can be set from 0 to max_gene_value (both inclusive)<BR>
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvKnaryVectorSpace implements EvSolutionSpace<EvKnaryIndividual> {

  private static final long serialVersionUID = 4094679815783435136L;

  // number of genes
  private int dimension;

  private int max_gene_value;

  private EvObjectiveFunction<EvKnaryIndividual> objective_function;


  /**
   * Creates solution space that generates individuals of given parameters.
   * 
   * @param dimension - number of genes in individual
   * @param max_gene_value
   */
  public EvKnaryVectorSpace(int dimension, int max_gene_value) {
    this.dimension = dimension;
    this.max_gene_value = max_gene_value;
  }


  /**
   * Gets dimension (length) of individuals chromosome
   * 
   * @return dimension of individual
   */
  public int getDimension() {
    return dimension;
  }


  /**
   * Check if individual given with argument <code>individual</code> belongs
   * to this solution space. True if individual has the same maximum gene value
   * and has the same dimension.
   * 
   * @return if individual belongs to this solution space
   */
  public boolean belongsTo(EvKnaryIndividual individual) {
    if (individual == null) {
      return false;
    }
    if (individual.getMaxGeneValue() != max_gene_value) {
      return false;
    } else {
      return individual.getDimension() == this.dimension;
    }
  }


  /**
   * Generates random EvKnaryIndividual of dimension, maximum gene value
   * specified and objective function in the solution space.<BR>
   * Each gene is shuffled randomly from 0 to max_gene_value (inclusively)<BR>
   * 
   * @return new random binary individual
   */
  public EvKnaryIndividual generateIndividual() {

    EvKnaryIndividual individual =
        new EvKnaryIndividual(dimension, max_gene_value);
    int individual_size = individual.getDimension();
    for (int i = 0; i < individual_size; i++) {
      individual.setGene(i, EvRandomizer.INSTANCE.nextInt(max_gene_value + 1)); // randomize
                                                                                // 0 or
                                                                                // 1
    }
    individual.setObjectiveFunction(objective_function);
    return individual;
  }


  /**
   * {@inheritDoc}
   */
  public void setObjectiveFuntion(
      EvObjectiveFunction<EvKnaryIndividual> objective_function) {
    if (objective_function == null) {
      throw new IllegalArgumentException("Objective function cannot be null");
    }
    this.objective_function = objective_function;
  }


  /**
   * {@inheritDoc}
   */
  public EvObjectiveFunction<EvKnaryIndividual> getObjectiveFuntion() {
    return objective_function;
  }


  /**
   * If only individual is from correct class and dimension, return this
   * individual itself. In other case return null.
   * 
   * @return binary individual itself or null
   */
  public EvKnaryIndividual takeBackTo(EvKnaryIndividual individual) {
    if (this.belongsTo(individual))
      return individual;
    else
      return null;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvKnaryIndividual>> divide(int n) {
    return null;
  }


  /**
   * [not used in current version]
   * 
   * @return [nil]
   */
  public Set<EvSolutionSpace<EvKnaryIndividual>> divide(int n,
      Set<EvKnaryIndividual> p) {
    return null;
  }

}
