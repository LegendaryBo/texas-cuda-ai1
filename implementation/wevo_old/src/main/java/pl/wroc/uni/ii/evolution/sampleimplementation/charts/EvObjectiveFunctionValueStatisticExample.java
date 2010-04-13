package pl.wroc.uni.ii.evolution.sampleimplementation.charts;

import java.io.IOException;

import org.jfree.chart.JFreeChart;

import com.sun.image.codec.jpeg.ImageFormatException;

import pl.wroc.uni.ii.evolution.chart.EvChartTools;
import pl.wroc.uni.ii.evolution.chart.EvObjectiveFunctionValueMaxAvgMinChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticSerializationStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorOnePointCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvSGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

public class EvObjectiveFunctionValueStatisticExample {

  /**
   * Example of use OneMax (and also GenericEvolutionaryAlgorithm).
   * 
   * @param no default parameters
   * @throws IOException
   * @throws ImageFormatException
   */
  public static void main(String[] args) throws ImageFormatException,
      IOException {

    EvPersistentStatisticStorage storage =
        new EvPersistentStatisticSerializationStorage("C:/wevo/");
    EvTask evolutionary_task = new EvTask();

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

    sga.setSolutionSpace(new EvBinaryVectorSpace(new EvOneMax(), 100));
    sga
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));
    sga
        .addOperatorToEnd(new EvObjectiveFunctionValueMaxAvgMinGatherer<EvBinaryVectorIndividual>(
            storage));
    sga.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
        300));
    evolutionary_task.setAlgorithm(sga);
    evolutionary_task.run();

    EvStatistic[] stats =
        EvStatisticFilter.byClass(
            EvObjectiveFunctionValueMaxAvgMinStatistic.class, storage
                .getStatistics());
    JFreeChart chart =
        EvObjectiveFunctionValueMaxAvgMinChart.createJFreeChart(stats, true);

    // put statistics into an image
    EvChartTools.createJPG(chart, "c:\\max_avg_min.jpg", 300, 600, 1);

  }

}
