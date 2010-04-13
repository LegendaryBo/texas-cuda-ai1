package pl.wroc.uni.ii.evolution.engine.prototype;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvBinaryPattern;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

public class EvIndividualTest extends TestCase {

  private class TestingIndividual extends EvIndividual {
    private static final long serialVersionUID = 1L;

    public int value;

    /* Constructor which sets objective function. */
    public TestingIndividual(int n) {
      value = n;
      setObjectiveFunction(new EvObjectiveFunction<TestingIndividual>() {

        private static final long serialVersionUID = -3449823855246165275L;

        public double evaluate(TestingIndividual individual) {
          return value;
        }
      });
    }
    
    public TestingIndividual clone() {
      return new TestingIndividual(value);
    }
  }

  /** Tests if individual.compareTo(another_individual) working correct. */
  public void testIndividualComparison() {
    TestingIndividual individual1 = new TestingIndividual(69);
    TestingIndividual individual2 = new TestingIndividual(102);

    assertEquals(69.0, individual1.getObjectiveFunctionValue(), 0.01);
    assertTrue(individual1.compareTo(individual2) < 0);
  }

  public void testAutomaticInvalidate() throws Exception {
    EvBinaryVectorIndividual individual = new EvBinaryVectorIndividual(new int[]{1, 0});
    individual.setObjectiveFunction(new EvOneMax());
    double goal_func_value = individual.getObjectiveFunctionValue();
    assertEquals(1.0,goal_func_value);
    
    individual.setObjectiveFunction(new EvBinaryPattern(new int[] {1, 0}));
    assertEquals(2.0,individual.getObjectiveFunctionValue());
    
    assertFalse(goal_func_value == individual.getObjectiveFunctionValue());
  }
}
