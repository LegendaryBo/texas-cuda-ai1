package pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genesorigin;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

/**
 * EvStatistic object which contains information about origin of the gene
 * statistics of the single iteration. It can store best individuals in
 * specified iteration of number of genes discovered in this iteration
 * 
 * @author Kacper Gorski
 */
public class EvGenesOriginStatistic extends EvStatistic {

  private static final long serialVersionUID = -3133527973605544762L;

  // discovered genes
  public List<Double>[] genes_discovered;

  // best individual in this iteration
  public double[] best_genes;


  public EvGenesOriginStatistic(int iteration, List<Double>[] genes_discovered,
      double[] best_genes) {
    this.genes_discovered = genes_discovered;
    this.best_genes = best_genes;
    this.setIteration(iteration);
  }

}
