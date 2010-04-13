package pl.wroc.uni.ii.evolution.sampleimplementation.charts;

import java.io.IOException;
import org.jfree.chart.JFreeChart;
import pl.wroc.uni.ii.evolution.chart.EvChartTools;
import pl.wroc.uni.ii.evolution.chart.EvGenesChangesChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoSelectionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticFileStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorOnePointCrossover;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.EvBinaryVectorGenesChangesGatherer;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvSGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import com.sun.image.codec.jpeg.ImageFormatException;

public class EvBinaryVectorGenesChangeStatisticExample {
  public static void main(String[] args) throws ImageFormatException,
      IOException {

    EvPersistentStatisticStorage storage =
        new EvPersistentStatisticFileStorage("C:/ecga_gene_changes");

    EvOneMax objective_function = new EvOneMax();

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

    sga.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 30));
    sga.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
        1000));

    sga.addOperatorToEnd(new EvBinaryVectorGenesChangesGatherer(storage));
    long cur_time = System.currentTimeMillis();

    EvTask evolutionary_task = new EvTask();
    evolutionary_task.setAlgorithm(sga);

    evolutionary_task.run();
    long after_time = System.currentTimeMillis();
    System.out
        .println("RUN TIME: " + ((after_time - cur_time) / 1000) + " sec");

    int[] genes = new int[40];
    for (int i = 0; i < genes.length; i++) {
      genes[i] = i;
    }

    EvStatistic[] stats = storage.getStatistics();

    JFreeChart chart = EvGenesChangesChart.createJFreeChart(stats, true);

    EvChartTools.createJPG(chart, "C:/binary_gene_changes.jpg", 500, 1000, 100);

  }
}
