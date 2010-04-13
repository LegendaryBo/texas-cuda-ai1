package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.statistic;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genesorigin.EvGenesOriginStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

/**
 * @author Marcin Golebiowski
 */
public class EvRealVectorGenesOriginGatherer extends
    EvGatherer<EvRealVectorIndividual> {

  private int iteration = 0;

  private TreeSet<Double>[] discovered_genes;


  public EvRealVectorGenesOriginGatherer(EvPersistentStatisticStorage storage) {
    setStorage(storage);
  }


  @SuppressWarnings( {"unchecked", "unchecked"})
  @Override
  public EvStatistic generate(EvPopulation<EvRealVectorIndividual> population) {

    // dimension of natural_number_vector_individual
    int dimension = population.get(0).getDimension();

    // create structure for discovered genes in previous iteration if it don't
    // exist
    if (discovered_genes == null) {
      this.discovered_genes = new TreeSet[dimension];

      for (int i = 0; i < dimension; i++) {
        this.discovered_genes[i] = new TreeSet<Double>();
      }
    }

    // create structure for new genes
    List<Double>[] new_genes = new List[dimension];
    for (int i = 0; i < dimension; i++) {
      new_genes[i] = new ArrayList<Double>();
    }

    // decide if gene is new or old
    for (EvRealVectorIndividual individual : population) {

      for (int i = 0; i < individual.getDimension(); i++) {
        if (!discovered_genes[i].contains(individual.getValue(i))) {
          new_genes[i].add((double) individual.getValue(i));
          discovered_genes[i].add((double) individual.getValue(i));
        }
      }
    }

    // get best individual's genes
    EvRealVectorIndividual best_individual = population.getBestResult();
    double best_genes[] = new double[dimension];
    for (int i = 0; i < dimension; i++) {
      best_genes[i] = (double) best_individual.getValue(i);
    }

    // create statistic
    EvStatistic stats =
        new EvGenesOriginStatistic(iteration++, new_genes, best_genes);

    // return result
    return stats;

  }

}
