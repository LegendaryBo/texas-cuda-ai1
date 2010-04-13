package pl.wroc.uni.ii.evolution.objectivefunctions;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKPattern;
import junit.framework.TestCase;

public class EvKPatternTest  extends TestCase {
  
  public void testCompute() {
    
    EvBinaryVectorIndividual individual = 
      new EvBinaryVectorIndividual(new int[] {1, 0, 1, 1, 1});
    EvKPattern obj = new EvKPattern(new int[] {1, 0}, 5);
    assertEquals((5.0 + 1.0) + (2.0)  + (1.0), obj.evaluate(individual)); 
    
    
    EvBinaryVectorIndividual individual2 = 
      new EvBinaryVectorIndividual(new int[] {1, 0, 0, 1, 1, 0});
    EvKPattern obj2 = new EvKPattern(new int[] {1, 0, 0}, 5);
    assertEquals((5.0 + 1.0) + (2.0), obj2.evaluate(individual2)); 
    
    
  }

}
