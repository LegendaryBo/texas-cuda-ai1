package pl.wroc.uni.ii.evolution.sampleimplementation.comparision;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorBOAOperator;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvSGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.Ev3Deceptive;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

public class EvSBOAStrategyBench {

  public static void main(String[] args) {

    int d = 60;
    int k = 2;
    int n = 4000;
    int iter = 20;

    long start = System.currentTimeMillis();

    EvSGA<EvBinaryVectorIndividual> sga =
        new EvSGA<EvBinaryVectorIndividual>(n,
            new EvKBestSelection<EvBinaryVectorIndividual>(n / 2),
            new EvBinaryVectorBOAOperator(d, k, n / 2),
            new EvBestFromUnionReplacement<EvBinaryVectorIndividual>());

    sga.setSolutionSpace(new EvBinaryVectorSpace(new Ev3Deceptive(), d));
    sga
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));
    sga.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
        iter));
    sga.init();
    sga.run();

    long end = System.currentTimeMillis();

    System.out.println(end - start + "ms");

  }

}
