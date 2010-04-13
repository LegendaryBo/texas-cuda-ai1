package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorCGAOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvBestValueNotImproved;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace;

/**
 * An example of modification of CGA to vector of real numbers. Very lightweight
 * algorithm to optimize real number functions.
 * 
 * @author Marcin Brodziak
 */
public class EvRealCGAExample {

  /**
   * @param args
   */
  @SuppressWarnings("unchecked")
  public static void main(String[] args) {

    EvTask evolutionary_task = new EvTask();

    EvAlgorithm<EvRealVectorIndividual> genericEA;

    int size = 10;

    genericEA = new EvAlgorithm<EvRealVectorIndividual>(2);
    genericEA.setSolutionSpace(new EvRealVectorSpace(new EvRealOneMax(), size));
    genericEA.setObjectiveFunction(new EvRealOneMax());

    genericEA.addOperatorToEnd(new EvRealVectorCGAOperator(size, 0.02, 0.002));

    genericEA.setTerminationCondition(new EvBestValueNotImproved(85));
    genericEA
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvRealVectorIndividual>(
            System.out));

    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();
  }

}
