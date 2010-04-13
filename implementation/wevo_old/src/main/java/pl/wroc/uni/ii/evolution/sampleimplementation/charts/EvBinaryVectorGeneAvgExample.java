package pl.wroc.uni.ii.evolution.sampleimplementation.charts;

import java.io.IOException;

import org.jfree.chart.JFreeChart;

import pl.wroc.uni.ii.evolution.chart.EvChartTools;
import pl.wroc.uni.ii.evolution.chart.EvGenesAverageValuesChart;
import pl.wroc.uni.ii.evolution.chart.EvObjectiveFunctionValueMaxAvgMinChart;
import pl.wroc.uni.ii.evolution.distribution.statistics.persistency.EvPersistentStatisticServletStorage;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveFunctionDistributionGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.statistic.EvBinaryGenesAvgValueGatherer;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvECGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

import com.sun.image.codec.jpeg.ImageFormatException;

public class EvBinaryVectorGeneAvgExample {
  public static void main(String[] args) throws ImageFormatException,
      IOException {

    Long task = 11L;
    Long cell = 111L;
    Long node = System.currentTimeMillis();

    String wevo_url = "http://127.0.0.1:8080";

    EvDBServletCommunicationImpl comm =
        new EvDBServletCommunicationImpl(wevo_url);

    // EvMySQLDirectCommunication comm = new
    // EvMySQLDirectCommunication("jdbc:mysql://127.0.0.1", "root", "n");

    EvPersistentStatisticStorage storage =
        new EvPersistentStatisticServletStorage(task, cell, node, comm);

    EvKDeceptiveOneMax objective_function = new EvKDeceptiveOneMax(4);
    EvAlgorithm<EvBinaryVectorIndividual> ecga = new EvECGA(false, 2000, 16, 4);

    ecga.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 40));
    ecga.setObjectiveFunction(objective_function);
    ecga.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(
        10));
    EvTask evolutionary_task = new EvTask();
    evolutionary_task.setAlgorithm(ecga);
    ecga
        .addOperatorToEnd(new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));

    ecga.addOperatorToEnd(new EvBinaryGenesAvgValueGatherer(40, storage));
    ecga
        .addOperatorToEnd(new EvObjectiveFunctionDistributionGatherer<EvBinaryVectorIndividual>(
            storage));
    ecga
        .addOperatorToEnd(new EvObjectiveFunctionValueMaxAvgMinGatherer<EvBinaryVectorIndividual>(
            storage));

    long cur_time = System.currentTimeMillis();
    evolutionary_task.run();
    long after_time = System.currentTimeMillis();
    System.out
        .println("RUN TIME: " + ((after_time - cur_time) / 1000) + " sec");

    int[] genes = new int[40];
    for (int i = 0; i < genes.length; i++) {
      genes[i] = i;
    }

    EvStatistic[] stats =
        EvStatisticFilter.byClass(EvGenesAvgValueStatistic.class, storage
            .getStatistics());
    EvStatistic[] stats2 =
        EvStatisticFilter.byClass(
            EvObjectiveFunctionValueMaxAvgMinStatistic.class, storage
                .getStatistics());
    JFreeChart chart = EvGenesAverageValuesChart.createJFreeChart(stats, true);
    JFreeChart chart2 =
        EvObjectiveFunctionValueMaxAvgMinChart.createJFreeChart(stats2, false);

    EvChartTools.createJPG(chart, "C:/binary_gene_avg.jpg", 500, 1000, 100);
    EvChartTools.createJPG(chart2, "C:/binary_minmaxavg.jpg", 500, 1000, 100);

  }
}
