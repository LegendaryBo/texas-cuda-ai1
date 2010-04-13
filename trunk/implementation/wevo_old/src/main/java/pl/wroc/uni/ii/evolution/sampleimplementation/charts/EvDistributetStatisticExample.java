package pl.wroc.uni.ii.evolution.sampleimplementation.charts;

import java.io.IOException;
import com.sun.image.codec.jpeg.ImageFormatException;

/*
 * import org.jfree.chart.JFreeChart; import
 * pl.wroc.uni.ii.evolution.chart.EvChartTools; import
 * pl.wroc.uni.ii.evolution.chart.EvGenesAverageValuesChart; import
 * pl.wroc.uni.ii.evolution.chart.EvObjectiveFunctionValueMaxAvgMinChart; import
 * pl.wroc.uni.ii.evolution.engine.EvAlgorithm; import
 * pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual; import
 * pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.general.statistic.avggenesvalue.EvGenesAvgValueStatistic;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinGatherer;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.general.statistic.objectivefunctiondistr.EvObjectiveFunctionDistributionGatherer;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticDatabaseSuppportServletStorage;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvStatisticFilter;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.spacespecific.binaryvector.statistic.EvBinaryGenesAvgValueGatherer;
 * import
 * pl.wroc.uni.ii.evolution.engine.operators.spacespecific.binaryvector.statistic.EvBinaryVectorGenesChangesGatherer;
 * import pl.wroc.uni.ii.evolution.engine.prototype.EvTask; import
 * pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvECGA; import
 * pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration; import
 * pl.wroc.uni.ii.evolution.objectivefunctions.EvKDeceptiveOneMax; import
 * pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
 * import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
 */
/**
 * Example algorithm that runs ecga and gather 4 kind statistics to a database
 * which can be read using EvDistributetStatisticsApplet. @ put task 11 to a
 * database first! @
 * @author Kacper Gorski
 */
public class EvDistributetStatisticExample {
  public static void main(String[] args) throws ImageFormatException,
      IOException {

    // Long task = 11L;
    // Long cell= 111L;
    // Long node = System.currentTimeMillis();
    //   
    //   
    //   
    // //EvDBServletCommunicationImpl comm = new
    // EvDBServletCommunicationImpl(download, upload );
    //
    // //EvMySQLDirectCommunication comm = new
    // EvMySQLDirectCommunication("jdbc:mysql://127.0.0.1", "root", "n");
    //   
    // EvPersistentStatisticStorage storage =
    // new EvPersistentStatisticDatabaseSuppportServletStorage(task, cell, node,
    // comm);
    //    
    //    
    // EvKDeceptiveOneMax objective_function = new EvKDeceptiveOneMax(4);
    // EvAlgorithm<EvBinaryVectorIndividual> ecga = new EvECGA(false, 2000, 16,
    // 4);
    //      
    // ecga.setSolutionSpace(new EvBinaryVectorSpace(objective_function, 40));
    // ecga.setObjectiveFunction(objective_function);
    // ecga.setTerminationCondition(new
    // EvMaxIteration<EvBinaryVectorIndividual>(40));
    // EvTask evolutionary_task = new EvTask();
    // evolutionary_task.setAlgorithm(ecga);
    //    
    // ecga.addOperator(new EvBinaryGenesAvgValueGatherer(40, storage));
    // ecga.addOperator(new
    // EvObjectiveFunctionDistributionGatherer<EvBinaryVectorIndividual>(storage));
    // ecga.addOperator(new
    // EvObjectiveFunctionValueMaxAvgMinGatherer<EvBinaryVectorIndividual>(storage));
    // ecga.addOperator(new EvBinaryVectorGenesChangesGatherer(storage));
    //    
    // long cur_time = System.currentTimeMillis();
    // evolutionary_task.run();
    // long after_time = System.currentTimeMillis();
    // System.out.println("RUN TIME: " + ((after_time - cur_time)/1000) + "
    // sec");
    //    
    //    
    // int[] genes = new int[40];
    // for (int i = 0; i < genes.length; i++) {
    // genes[i] = i;
    // }
    //    

  }
}
