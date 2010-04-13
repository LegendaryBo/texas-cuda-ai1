package pl.wroc.uni.ii.evolution.objectivefunctions;

import pl.wroc.uni.ii.evolution.engine.individuals.EvPermutationIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.permutation.EvLessCycleNumber;
import junit.framework.TestCase;
/**
 * @author Donata Malecka, Piotr Baraniak
 */
public class EvLessCycleNumberTest extends TestCase {
  public void testIfItWorks() {
    EvLessCycleNumber function = new EvLessCycleNumber();
    EvPermutationIndividual individual = new EvPermutationIndividual(new int[]{1,2,3,0});
    assertEquals("One cycle",1.0,function.evaluate(individual));
    individual = new EvPermutationIndividual(new int[]{1,0,3,2});
    assertEquals("Two cycle",0.5,function.evaluate(individual));
    individual = new EvPermutationIndividual(new int[]{0,1,2,3});
    assertEquals("Four cycle",0.25,function.evaluate(individual));
  }
}
