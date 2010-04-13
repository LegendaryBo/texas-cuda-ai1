package pl.wroc.uni.ii.evolution.sampleimplementation.charts;

import java.io.IOException;
import org.jfree.chart.JFreeChart;
import com.sun.image.codec.jpeg.ImageFormatException;
import pl.wroc.uni.ii.evolution.chart.EvChartTools;
import pl.wroc.uni.ii.evolution.chart.EvGenesOriginChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvBlockSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticSerializationStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.EvNaturalNumberVectorChangeMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.EvNaturalNumberVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.statistic.EvNaturalNumberGenesOriginGatherer;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.naturalnumbervector.EvNaturalPattern;
import pl.wroc.uni.ii.evolution.solutionspaces.EvNaturalNumberVectorSpace;

/**
 * Simple example how to use EvGenesOriginGatherer to generate statistics
 * 
 * @author Kacper Gorski
 */
public class EvNaturalNumberVectorGenesOriginStatisticExample {

  public static void main(String[] args) throws ImageFormatException,
      IOException {

    EvPersistentStatisticStorage storage =
        new EvPersistentStatisticSerializationStorage("C:/wevo");

    EvAlgorithm<EvNaturalNumberVectorIndividual> genericEA = null;
    EvTask evolutionary_task = new EvTask();

    genericEA = new EvAlgorithm<EvNaturalNumberVectorIndividual>(100);
    genericEA.setSolutionSpace(new EvNaturalNumberVectorSpace(
        new EvNaturalPattern(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}), 10,
        300));
    genericEA
        .addOperatorToEnd(new EvBlockSelection<EvNaturalNumberVectorIndividual>(
            20));
    genericEA.addOperatorToEnd(new EvNaturalNumberVectorUniformCrossover());
    genericEA.addOperatorToEnd(new EvNaturalNumberVectorChangeMutation(0.2, 1));
    genericEA
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvNaturalNumberVectorIndividual>(
            System.out));

    genericEA
        .setTerminationCondition(new EvMaxIteration<EvNaturalNumberVectorIndividual>(
            100));

    genericEA.addOperatorToEnd(new EvNaturalNumberGenesOriginGatherer(storage));
    evolutionary_task.setAlgorithm(genericEA);
    evolutionary_task.run();
    evolutionary_task.printBestResult();

    EvStatistic[] stats = storage.getStatistics();

    JFreeChart chart = EvGenesOriginChart.createJFreeChart(stats);

    // put statistics into an image
    EvChartTools.createJPG(chart, "c:\\natural_number_origin.jpg", 300, 600, 1);

  }

}
