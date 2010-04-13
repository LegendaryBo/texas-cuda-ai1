package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.scatterplot;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * The reason this class exist is that there is a need to create scatter plots
 * for different types of individuals. Unfortunately we can't make a scatter
 * plot from every EvIndividual (only individuals, that has genes convertable
 * into double values, like EvKNaryIndividual, or EvRealVectorIndividual).
 * Operator implementing this class will need to define two simple
 * individual-specific functions:<br> - double getGene(T individual, index) -
 * get double value of the specified gene.<br> - double getNumberOfGenes(T
 * individual) - get number of genes in given individual.<br>
 * <br>
 * <br>
 * NOTE: You must call one of two main constructors in this class if you want
 * your operator to work correctly.
 * 
 * @author Kacper Gorski (admin@34all.org)
 * @param <T> - type of individual about thich the operator will collect
 *        informations.
 */
public abstract class EvAbstractScatterPlotGatherer<T extends EvIndividual>
    implements EvOperator<T> {

  /**
   * This method shall return double value of gene with specified index.
   * 
   * @param individual containing the gene
   * @param index of the gene
   * @return double value of the gene
   */
  public abstract double getGene(T individual, int index);


  /**
   * This method shall return number of genes in the given individual.
   * 
   * @param individual - sample individual
   * @return number of genes in given individual
   */
  public abstract int getNumberOfGenes(T individual);

  /**
   * {@inheritDoc}
   */
  private int iteration = 0;

  /**
   * Stores genes indexes that are to be stored.
   */
  private int[] genes_indexes = null;

  /**
   * Class in which we store data.
   */
  private EvPersistentStatisticStorage storage = null;


  /**
   * Default constructor, it saves info of all genes.
   * 
   * @param stor in which statistics are stored
   */
  public EvAbstractScatterPlotGatherer(final EvPersistentStatisticStorage stor) {
    this.storage = stor;
  }


  /**
   * Creates operator that will store genes specified in genes_ind table.<br>
   * NOTE: first gene starts with index 0
   * 
   * @param genes_ind table containing indexes of genes about which informations
   *        going to be gathered
   * @param stor - storage in which statistics are stored
   */
  public EvAbstractScatterPlotGatherer(final int[] genes_ind,
      final EvPersistentStatisticStorage stor) {
    this.storage = stor;
    this.genes_indexes = genes_ind;
  }


  /**
   * Collect data about current population. It does not affect the population in
   * any way
   * 
   * @param population about which data is collected
   * @return same population
   */
  public EvPopulation<T> apply(final EvPopulation<T> population) {

    // genes_indexes wasn't init, so we gonna store all genes
    if (genes_indexes == null && population.get(0) != null) {
      genes_indexes = new int[getNumberOfGenes(population.get(0))];
      for (int i = 0; i < getNumberOfGenes(population.get(0)); i++) {
        genes_indexes[i] = i;
      }
    }

    EvScatterPlotStatistic stats =
        new EvScatterPlotStatistic(copySelectedGenes(population), iteration);
    iteration++;

    storage.saveStatistic(stats);

    return population;
  }


  /**
   * @param population whose individuals' genes will be copied
   * @return table containing specified genes of all individuals in given
   *         population.<br>
   *         Each individual in each row of a table
   */
  private double[][] copySelectedGenes(final EvPopulation<T> population) {

    double[][] genes = new double[population.size()][genes_indexes.length];

    for (int i = 0; i < population.size(); i++) {

      for (int j = 0; j < genes_indexes.length; j++) {
        genes[i][j] = getGene(population.get(i), genes_indexes[j]);
      }

    }

    return genes;
  }

}
