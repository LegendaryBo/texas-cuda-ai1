package pl.wroc.uni.ii.evolution.objectivefunctions;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.naturalnumbervector.EvNaturalPattern;

public class EvNaturalPatternTest extends TestCase {
  public void testNaturalPatternTest() throws Exception {
    EvNaturalNumberVectorIndividual individual = new EvNaturalNumberVectorIndividual(new int[] {1,2,3,4});
    individual.setObjectiveFunction(new EvNaturalPattern(new int[] {1,2,3,4}));
    assertEquals(4, individual.getObjectiveFunctionValue(),0.01);
    individual.setNumberAtPosition(1, 7);
    assertEquals(3,individual.getObjectiveFunctionValue(),0.01);
  }
}
