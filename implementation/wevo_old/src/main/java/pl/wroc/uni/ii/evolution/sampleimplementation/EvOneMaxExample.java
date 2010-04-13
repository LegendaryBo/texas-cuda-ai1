package pl.wroc.uni.ii.evolution.sampleimplementation;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorOnePointCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorBOAOperator;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvCGA;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvSGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvBestValueNotImproved;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Example of use OneMax (and also GenericEvolutionaryAlgorithm).
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvOneMaxExample {

  /**
   * disabling constructor.
   */
  protected EvOneMaxExample() {
    throw new IllegalStateException("Class shouldn't be istantiated");
  }
  
  /**
   * 
   * 
   * @param args no default parameters.
   */
  public static void main(final String[] args) {

    EvOneMax objective_function = new EvOneMax();

    EvCGA cga = new EvCGA(0.04);
    cga.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 30));
    cga.setObjectiveFunction(objective_function);
    cga.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
        1500));

    EvTask evolutionary_task = new EvTask();
    evolutionary_task.setAlgorithm(cga);
    evolutionary_task.run();
    evolutionary_task.printBestResult();

    EvAlgorithm<EvBinaryVectorIndividual> genericEA =
        new EvAlgorithm<EvBinaryVectorIndividual>(100);
    genericEA.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 30));
    genericEA.setObjectiveFunction(objective_function);
    genericEA.addOperatorToEnd(new EvKBestSelection<EvBinaryVectorIndividual>(
        50));
    genericEA.addOperatorToEnd(
        new EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual>());
    
    genericEA.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(
            100));

    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();

    EvBinaryVectorBOAOperator boa = 
        new EvBinaryVectorBOAOperator(4, 10, 1000);
    
    genericEA = new EvAlgorithm<EvBinaryVectorIndividual>(1000);
    genericEA.setSolutionSpace(
        new EvBinaryVectorSpace(objective_function, 30));
    genericEA.setObjectiveFunction(new EvKDeceptiveOneMax(4));
    genericEA.addOperatorToEnd(
        new EvReplacementComposition<EvBinaryVectorIndividual>(
            boa,
            new EvBestFromUnionReplacement<EvBinaryVectorIndividual>()));

    genericEA.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(1));

    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();

    System.out.println("--==SGA==---");

    EvAlgorithm<EvBinaryVectorIndividual> sga =
        new EvSGA<EvBinaryVectorIndividual>(
            100,
            new EvTwoSelectionComposition<EvBinaryVectorIndividual>(
                new EvRandomSelection<EvBinaryVectorIndividual>(16, false),
                new EvKBestSelection<EvBinaryVectorIndividual>(4)),
            new EvTwoOperatorsComposition<EvBinaryVectorIndividual>(
                new EvBinaryVectorNegationMutation(0.02),
               new EvKnaryVectorOnePointCrossover<EvBinaryVectorIndividual>()),
            new EvBestFromUnionReplacement<EvBinaryVectorIndividual>());

    sga.setSolutionSpace(new EvBinaryVectorSpace(new EvOneMax(), 30));
    sga.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));
    sga.setTerminationCondition(
        new EvBestValueNotImproved<EvBinaryVectorIndividual>(
            4));
    evolutionary_task.setAlgorithm(sga);
    evolutionary_task.run();
    evolutionary_task.printBestResult();
  }

}
