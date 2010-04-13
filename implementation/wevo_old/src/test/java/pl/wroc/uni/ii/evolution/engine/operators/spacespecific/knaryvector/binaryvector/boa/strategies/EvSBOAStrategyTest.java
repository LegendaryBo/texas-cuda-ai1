package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.strategies;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.boa.metrics.EvBoaStrategy;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.Ev3Deceptive;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import junit.framework.TestCase;

public class EvSBOAStrategyTest extends TestCase {

  
  public void testComputeMetric2() {
    
  /*
      int d = 30;
      int n = 2000;
      
      EvBinaryVectorSpace space = new EvBinaryVectorSpace(new Ev3Deceptive(), d);
      EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>();
    
      for (int i = 0; i < n; i++) {
        population.add(space.generateIndividual());
      }
    
      EvBoaStrategy strategy = new EvBoaStrategy();
   
      strategy.init(population.size() + 1);
      //strategy.
      double val1 = strategy.computeMetric3(0, new int[] {1, 2}, population.toArray(new EvBinaryVectorIndividual[n]));
      double val2 = strategy.computeMetric2(0, new int[] {1, 2}, population.toArray(new EvBinaryVectorIndividual[n]));
      double val3 = strategy.computeMetric(0, new int[] {1, 2}, population.toArray(new EvBinaryVectorIndividual[n]));
      
      System.out.println(val1);
      System.out.println(val2);
      System.out.println(val3); */
  }
  
  public void testMetrics() {
    
    System.out.println("START");
    
    int d = 5;
    int n = 2000;
    
    EvBinaryVectorSpace space = new EvBinaryVectorSpace(new Ev3Deceptive(), d);
    EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>();
  
    for (int i = 0; i < n; i++) {
      population.add(space.generateIndividual());
    }
  
    EvBoaStrategy strategy = new EvBoaStrategy();
    strategy.init(population.size() + 1, 3);
    
    double[] result =  strategy.computeMetics(0, new int[] {1, 3}, new boolean[] {false, false, true, false}, 
        population.toArray(new EvBinaryVectorIndividual[population.size()]));
   
    double val1 = strategy.computeMetric3(0, new int[] {1, 2, 3}, population.toArray(new EvBinaryVectorIndividual[n]));
    
    assertEquals(4, result.length);
   
    
    //for (int i = 0; i < result.length; i++) {
    //  System.out.println(result[i]);
    //}
    
    // numerical error
    double deviation = 0.00000000001;
    
    assertTrue((val1 + deviation > result[2]) && (val1 - deviation < result[2]));
    
    
    System.out.println("END");
  }
 

}
