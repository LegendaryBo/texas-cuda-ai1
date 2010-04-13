package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.statistic;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genesorigin.EvGenesOriginStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

/**
 * @author Marcin Golebiowski
 */
public class EvNaturalNumberGenesOriginGatherer extends
    EvGatherer<EvNaturalNumberVectorIndividual> {

  private int iteration = 0;

  private TreeSet<Double>[] discovered_genes;


  public EvNaturalNumberGenesOriginGatherer(EvPersistentStatisticStorage storage) {
    setStorage(storage);
  }


  @SuppressWarnings("unchecked")
  @Override
  public EvStatistic generate(
      EvPopulation<EvNaturalNumberVectorIndividual> population) {

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
    for (EvNaturalNumberVectorIndividual individual : population) {

      for (int i = 0; i < individual.getDimension(); i++) {
        if (!discovered_genes[i].contains((double) individual
            .getNumberAtPosition(i))) {
          new_genes[i].add((double) individual.getNumberAtPosition(i));
          discovered_genes[i].add((double) individual.getNumberAtPosition(i));
        }
      }
    }

    // get best individual's genes
    EvNaturalNumberVectorIndividual best_individual =
        population.getBestResult();
    double best_genes[] = new double[dimension];
    for (int i = 0; i < dimension; i++) {
      best_genes[i] = (double) best_individual.getNumberAtPosition(i);
    }

    // create statistic
    EvStatistic stats =
        new EvGenesOriginStatistic(iteration++, new_genes, best_genes);

    // return result
    return stats;

  }

}
