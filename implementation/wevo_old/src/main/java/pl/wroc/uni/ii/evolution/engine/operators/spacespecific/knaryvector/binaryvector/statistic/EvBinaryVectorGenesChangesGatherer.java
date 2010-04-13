package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genechange.EvGeneChangeStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;

public class EvBinaryVectorGenesChangesGatherer extends
    EvGatherer<EvBinaryVectorIndividual> {

  private boolean[] previous_best_vector = null;

  private int iteration = 0;


  public EvBinaryVectorGenesChangesGatherer(EvPersistentStatisticStorage storage) {
    setStorage(storage);
  }


  @Override
  public EvStatistic generate(EvPopulation population) {

    EvBinaryVectorIndividual current_best =
        (EvBinaryVectorIndividual) population.getBestResult();

    if (previous_best_vector == null) {

      previous_best_vector = getChromosome(current_best);
      return new EvGeneChangeStatistic(iteration++, 0);

    } else {

      boolean[] current_best_vector = getChromosome(current_best);

      int number_of_changes = 0;
      for (int i = 0; i < current_best.getDimension(); i++) {
        if (current_best_vector[i] != previous_best_vector[i]) {
          number_of_changes++;
        }
      }

      previous_best_vector = current_best_vector;
      return new EvGeneChangeStatistic(iteration++, number_of_changes);
    }
  }


  private boolean[] getChromosome(EvBinaryVectorIndividual individual) {
    boolean[] result = null;
    result = new boolean[individual.getDimension()];
    for (int i = 0; i < individual.getDimension(); i++) {
      result[i] = individual.getGene(i) == 1;
    }
    return result;
  }

}
