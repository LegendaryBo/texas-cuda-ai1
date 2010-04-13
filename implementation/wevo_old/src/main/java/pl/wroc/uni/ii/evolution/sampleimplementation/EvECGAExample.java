package pl.wroc.uni.ii.evolution.sampleimplementation;

import javax.swing.JFrame;

import pl.wroc.uni.ii.evolution.chart.EvBayesianChart;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvRealtimeToPrintStreamStatistics;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorECGAOperator;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * Example of ECGA algorithm.
 * Its shows progress and the way building blocks work
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvECGAExample {

  /**
   * Disabling constructor.
   */
  protected EvECGAExample() {
    throw new IllegalStateException("Class cannot be instantiated");
  }
  
  /**
   * 
   * 
   * @param args - not used
   */
  public static void main(final String[] args) {

    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
    
    EvKDeceptiveOneMax objective_function = new EvKDeceptiveOneMax(4);

    EvAlgorithm<EvBinaryVectorIndividual> alg = 
      new EvAlgorithm<EvBinaryVectorIndividual>(2000);

    EvBinaryVectorSpace solution = 
      new EvBinaryVectorSpace(objective_function, 60);
    
    EvBinaryVectorECGAOperator ecgaOper = 
      new EvBinaryVectorECGAOperator(4, false, solution);
    
    alg.setSolutionSpace(solution);
    alg.setObjectiveFunction(objective_function);
    alg.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(15));

    alg.addOperator(ecgaOper);

    ecgaOper.collectBayesianStats(storage);
    
    alg.addOperatorToEnd(
        new EvRealtimeToPrintStreamStatistics<EvBinaryVectorIndividual>(
            System.out));
    
    long cur_time = System.currentTimeMillis();
    alg.init();
    alg.run();
    long after_time = System.currentTimeMillis();
    
    System.out.println("RUN TIME: " + (after_time - cur_time) + " msec");
    
    
    
    // lets show some charts
    JFrame f = EvBayesianChart.createFrame(storage.getStatistics());
    

    f.pack();   
    f.setSize(700, 700); // add 20, seems enough for the Frame title,
    f.show(); 
    
    
    
    
  }  
  
}
