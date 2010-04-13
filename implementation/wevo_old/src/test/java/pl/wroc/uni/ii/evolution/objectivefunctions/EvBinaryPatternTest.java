package pl.wroc.uni.ii.evolution.objectivefunctions;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvBinaryPattern;

public class EvBinaryPatternTest extends TestCase {
  public void testBinaryPatternTest() throws Exception {
    EvBinaryVectorIndividual individual = new EvBinaryVectorIndividual(new int[] { 1,
        0, 0, 1, 0 });
    individual.setObjectiveFunction(new EvBinaryPattern(new int[] { 1,
        0, 0, 1, 0 }));
    assertEquals(5.0,individual.getObjectiveFunctionValue(),0.01);
    individual.setObjectiveFunction(new EvBinaryPattern(new int[] { 1,
        0, 1, 1, 0 }));
    assertEquals(4.0,individual.getObjectiveFunctionValue(),0.01);
  }
}
