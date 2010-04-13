package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.scatterplot;

import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

/**
 * Operator that collects data of specified genes in the population of
 * EvKnaryIndividuals and saves them in specified storage. <br>
 * This data can be later used to create scatter plots (pair of genes showed on
 * 2 - dimensional chart) It does nothing to the population.
 * 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvKNaryScatterPlotGatherer extends
    EvAbstractScatterPlotGatherer<EvKnaryIndividual> {

  /**
   * Default constructor, it saves info of all genes.
   * 
   * @param stor in which statistics are stored
   */
  public EvKNaryScatterPlotGatherer(final EvPersistentStatisticStorage stor) {
    super(stor);
  }


  /**
   * Creates operator that will store genes specified in genes_ind table.<br>
   * NOTE: first gene starts with index 0
   * 
   * @param genes_ind table containing indexes of genes about which informations
   *        going to be gathered
   * @param stor - storage in which statistics are stored
   */
  public EvKNaryScatterPlotGatherer(final int[] genes_ind,
      final EvPersistentStatisticStorage stor) {
    super(genes_ind, stor);
  }


  /**
   * {@inheritDoc}
   */
  public double getGene(final EvKnaryIndividual individual, final int index) {
    return individual.getGene(index);
  }


  /**
   * {@inheritDoc}
   */
  public int getNumberOfGenes(final EvKnaryIndividual individual) {
    return individual.getDimension();
  }

}
