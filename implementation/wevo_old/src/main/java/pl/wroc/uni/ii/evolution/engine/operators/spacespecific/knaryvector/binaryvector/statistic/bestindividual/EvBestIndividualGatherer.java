package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.bestindividual;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * @author Tomasz Kozakiewicz
 */
public class EvBestIndividualGatherer<T extends EvIndividual> extends
    EvGatherer<T> {
  // TODO generalize or change name
  protected int iteration = 0;


  public EvBestIndividualGatherer(EvPersistentStatisticStorage storage) {
    this.setStorage(storage);
  }


  @Override
  public EvStatistic generate(EvPopulation<T> population) {

    EvBinaryVectorIndividual individual =
        (EvBinaryVectorIndividual) population.getBestResult();
    return new EvBestIndividualStatistic(individual.getGenes(), individual
        .getObjectiveFunctionValue(), iteration++);
  }
}
