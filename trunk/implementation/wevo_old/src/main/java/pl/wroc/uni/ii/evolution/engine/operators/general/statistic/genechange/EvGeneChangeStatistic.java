package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genechange;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * Statistic for computation rate of changes of genes.
 * 
 * @author Marcin Golebiewski (xormus@gmail.com)
 */
public class EvGeneChangeStatistic extends EvStatistic {

  /** Serialization id. */
  private static final long serialVersionUID = 707528827719086092L;

  /** Number of changed genes. */
  private int number_of_genes_changed;


  /**
   * Creates statistic for computation of rate of change of genes.
   * 
   * @param iteration Iteration number.
   * @param number_of_genes_changed_p Number of genes that changed.
   */
  public EvGeneChangeStatistic(final int iteration,
      final int number_of_genes_changed_p) {
    setIteration(iteration);
    number_of_genes_changed = number_of_genes_changed_p;
  }


  /**
   * Returns number of genes that were changed.
   * 
   * @return Number of genes that were changed.
   */
  public int getNumberOfGenesChanged() {
    return number_of_genes_changed;
  }
}
