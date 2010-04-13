package pl.wroc.uni.ii.evolution.sampleimplementation.charts;

import java.io.IOException;
import org.jfree.chart.JFreeChart;
import com.sun.image.codec.jpeg.ImageFormatException;
import pl.wroc.uni.ii.evolution.chart.EvChartTools;
import pl.wroc.uni.ii.evolution.chart.EvGenesOriginChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvApplyOnSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvIterateCompositon;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticSerializationStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorAverageCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.EvRealVectorStandardDeviationMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.statistic.EvRealVectorGenesOriginGatherer;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvRealVectorSpace;

/**
 * Simple example how to use EvGenesOriginGatherer to generate statistics
 * 
 * @author Kacper Gorski
 */
public class EvRealVectorGenesOriginStatisticExample {

  public static void main(String[] args) throws ImageFormatException,
      IOException {

    EvPersistentStatisticStorage storage =
        new EvPersistentStatisticSerializationStorage("C:/wevo");

    EvTask evolutionary_task = new EvTask();

    EvAlgorithm<EvRealVectorIndividual> genericEA =
        new EvAlgorithm<EvRealVectorIndividual>(1000);
    genericEA.setSolutionSpace(new EvRealVectorSpace(
        new EvRealOneMax<EvRealVectorIndividual>(), 30));
    genericEA.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
    genericEA
        .setTerminationCondition(new EvMaxIteration<EvRealVectorIndividual>(20));
    genericEA
        .addOperatorToEnd(new EvReplacementComposition<EvRealVectorIndividual>(
            new EvIterateCompositon<EvRealVectorIndividual>(
                new EvApplyOnSelectionComposition<EvRealVectorIndividual>(
                    new EvTwoSelectionComposition<EvRealVectorIndividual>(
                        new EvRandomSelection<EvRealVectorIndividual>(16, false),
                        new EvKBestSelection<EvRealVectorIndividual>(4)),
                    new EvTwoOperatorsComposition<EvRealVectorIndividual>(
                        new EvRealVectorStandardDeviationMutation(0.01),
                        new EvRealVectorAverageCrossover(4)))),
            new EvBestFromUnionReplacement<EvRealVectorIndividual>()));

    // genericEA.addOperator(new
    // EvRealtimeToPrintStreamStatistics<EvRealVectorIndividual>(System.out));
    genericEA.addOperatorToEnd(new EvRealVectorGenesOriginGatherer(storage));

    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();

    EvStatistic[] stats = storage.getStatistics();

    JFreeChart chart = EvGenesOriginChart.createJFreeChart(stats);

    // put statistics into an image
    EvChartTools.createJPG(chart, "c:\\real_gene_origin.jpg", 300, 600, 1);

  }

}
