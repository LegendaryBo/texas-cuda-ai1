package pl.wroc.uni.ii.evolution.sampleimplementation.charts;

import javax.swing.JFrame;

import pl.wroc.uni.ii.evolution.chart.EvBayesianChart;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorBOAOperator;
import pl.wroc.uni.ii.evolution.engine.samplealgorithms.EvSGA;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.Ev3Deceptive;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * 
 * Simple example of charts wisualising bayesian network of sBOA operator.
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvBayesianNetworkChartExample {

  /**
   * Disabling constructor.
   */
  protected EvBayesianNetworkChartExample() {
    throw new UnsupportedOperationException();
  }  
  
  /**
   * @param args - not used
   */
  public static void main(final String[] args) {
   
    
    int genes = 15;
    int max_parents = 2;
    int n = 4000;
    int iter = 9;
    
    
    EvPersistentSimpleStorage storage = new EvPersistentSimpleStorage();
    EvBinaryVectorBOAOperator sboa = 
        new EvBinaryVectorBOAOperator(genes, max_parents, n / 2);
    
    sboa.collectBayesianStats(storage);
    
    EvSGA<EvBinaryVectorIndividual> sga = new EvSGA<EvBinaryVectorIndividual>(
        n, new EvKBestSelection<EvBinaryVectorIndividual>(n / 2),
        sboa,
        new EvBestFromUnionReplacement<EvBinaryVectorIndividual>());
    
    sga.setSolutionSpace(new EvBinaryVectorSpace(new Ev3Deceptive(), genes));

    sga.setTerminationCondition(
        new EvMaxIteration<EvBinaryVectorIndividual>(iter));
    sga.init();
    sga.run();    
    
    
    JFrame f = EvBayesianChart.createFrame(storage.getStatistics());


 
    
    f.pack();   
    f.setSize(800, 800); // add 20, seems enough for the Frame title,
    f.show(); 

  }

}
