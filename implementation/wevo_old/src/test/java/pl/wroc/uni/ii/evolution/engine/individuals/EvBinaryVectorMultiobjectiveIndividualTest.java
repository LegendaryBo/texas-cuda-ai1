/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.individuals;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvBinaryPattern;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

/**
 * Test for EvBinaryVectorMultiobjectiveIndividual.
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 *
 */
public class EvBinaryVectorMultiobjectiveIndividualTest {

  /** individual 1. */
  private EvBinaryVectorIndividual in1;
  /** individual 2. */
  private EvBinaryVectorIndividual in3;
  /** individual 3. */
  private EvBinaryVectorIndividual in2;
  /**
   * Setting up variables.
   * @throws java.lang.Exception Exception
   */
  @Before
  public void setUp() throws Exception {
    int[] pattern = {0, 0, 1, 1, 0};
    int[] genes1 = {1, 0, 1, 1, 0};
    int[] genes2 = {1, 1, 1, 1, 1};
    int[] genes3 = {0, 0, 0, 0, 0};
    EvBinaryPattern obj1 = new EvBinaryPattern(pattern);
    EvOneMax obj2 = new EvOneMax();
    in1 = new EvBinaryVectorIndividual(genes1);
    in2 = new EvBinaryVectorIndividual(genes2);
    in3 = new EvBinaryVectorIndividual(genes3);
    in1.addObjectiveFunction(obj1);
    in1.addObjectiveFunction(obj2);
    in2.addObjectiveFunction(obj1);
    in2.addObjectiveFunction(obj2);
    in3.addObjectiveFunction(obj1);
    in3.addObjectiveFunction(obj2);
  }

  /**
   * Test method compareTo.
   */
  @Test
  public void testCompareTo() {
    try {
      double v11 = in1.getObjectiveFunctionValue(0);
      double v12 = in1.getObjectiveFunctionValue(1);
      double v21 = in2.getObjectiveFunctionValue(0);
      double v22 = in2.getObjectiveFunctionValue(1);
      double v31 = in3.getObjectiveFunctionValue(0);
      double v32 = in3.getObjectiveFunctionValue(1);
      assertTrue("Objective fun. 1 for in1 has wrong value.", v11 == 4);
      assertTrue("Objective fun. 2 for in1 has wrong value.", v12 == 3);
      assertTrue("Objective fun. 1 for in2 has wrong value.", v21 == 2);
      assertTrue("Objective fun. 2 for in2 has wrong value.", v22 == 5);
      assertTrue("Objective fun. 1 for in3 has wrong value.", v31 == 3);
      assertTrue("Objective fun. 2 for in3 has wrong value.", v32 == 0);
      int c12 = in1.compareTo(in2);
      int c13 = in1.compareTo(in3);
      int c23 = in2.compareTo(in3);
      int c31 = in3.compareTo(in1);
      int c21 = in2.compareTo(in1);
      assertTrue("in1 should be equal to in2", c12 == 0);
      assertTrue("in1 should be higher than in3", c13 == 1);
      assertTrue("in2 should be equal to in3", c23 == 0);
      assertTrue("in3 should be smaller than in1", c31 == -1);
      assertTrue("in2 should be equal to in1", c21 == 0);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * Test method for isValidate(int).
   */
  @Test
  public void testIsValidateInt() {
    try {
      double v11 = in1.getObjectiveFunctionValue(0);
      double v12 = in1.getObjectiveFunctionValue(1);
      assertTrue("Objective fun. 1 for in1 should be validated.(A)", 
          in1.isEvaluated());
      assertTrue("Objective fun. 2 for in1 should be validated.(A)", 
          in1.isEvaluated());
      in1.setGene(1, 0);
      assertTrue("Objective fun. 1 for in1 shouldn't be validated.(B)", 
          !in1.isEvaluated());
      assertTrue("Objective fun. 2 for in1 shouldn't be validated.(B)", 
          !in1.isEvaluated());
      double val = in1.getObjectiveFunctionValue(0);
      assertTrue("Objective fun. 1 for in1 should be validated.(C)", 
          !in1.isEvaluated());
      assertTrue("Objective fun. 2 for in1 shouldn't be validated.(C)", 
          !in1.isEvaluated());
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

}
