package pl.wroc.uni.ii.evolution.engine.individuals;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;

/**
 * @author Kamil Dworakowski
 */
public class EvNaturalNumberIndividualTest extends TestCase {

  private EvNaturalNumberVectorIndividual individual;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    individual = new EvNaturalNumberVectorIndividual(new int[] { 3, 5, 6 });
  }
  
  public void testAnotherConstructor() throws Exception {
    EvNaturalNumberVectorIndividual indiv = new EvNaturalNumberVectorIndividual(3);
    assertEqualsVector(new int[] {0,0,0}, indiv);
  }

  public void testToString() throws Exception {
    assertEquals("3,5,6", individual.toString());
  }

  public void testGet() throws Exception {
    int[] expectedValues = new int[] {3,5,6};
    assertEqualsVector(expectedValues, individual);
  }
  
  public void testSet() throws Exception {
    individual.setNumberAtPosition(0,3);
    individual.setNumberAtPosition(1,2);
    individual.setNumberAtPosition(2,1);
    assertEqualsVector(new int[] {3,2,1},individual);
  }

  public static void assertEqualsVector(int[] expectedValues, EvNaturalNumberVectorIndividual individual) {
    assertEquals(expectedValues.length,individual.getDimension());
    for (int i = 0; i < expectedValues.length; i++)
      assertEquals(expectedValues[i],individual.getNumberAtPosition(i));
  }
}
