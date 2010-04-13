package pl.wroc.uni.ii.evolution.engine.individuals;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;

/**
 * @author Kamil Dworakowski, Jarek Fuks
 * 
 */
public class EvRealVectorIndividualTest extends TestCase {

  EvRealVectorIndividual individual;

  @Override
  protected void setUp() throws Exception {
    individual = new EvRealVectorIndividual(2);
    individual.setValue(0, 1.0d);
    individual.setValue(1, 2.0d);
  }

  public void testSetGet() throws Exception {

    assertEquals(1.0d, individual.getValue(0), 0.00000000001d);
    assertEquals(2.0d, individual.getValue(1), 0.00000000001d);
  }

  public void testGetDimensions() throws Exception {
    assertEquals(2, individual.getDimension());
  }

  public void testToString() throws Exception {
    assertEquals("1.0 2.0", individual.toString());
  }
  public void testCompare() throws Exception {

	    EvRealVectorIndividual individual1 = new EvRealVectorIndividual(new double[] {
	        0.5d, 0.5d });
	    EvRealVectorIndividual individual2 = new EvRealVectorIndividual(new double[] {
	        1.2d, 1.2d });
	    individual1.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
	    individual2.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());

	    assertEquals(1, individual2.compareTo(individual1));
	    assertEquals(-1, individual1.compareTo(individual2));
	    assertEquals(0, individual1.compareTo(individual1));
	  }

	  public void testAutomaticInvalidate() throws Exception {
	    EvRealVectorIndividual individual1 = new EvRealVectorIndividual(new double[] {
	        0.5d, 0.5d });

	    individual1.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());

	    double objective_function_value = individual1.getObjectiveFunctionValue();
	    individual1.setValue(0, 10d);
	    assertFalse(objective_function_value == individual1.getObjectiveFunctionValue());
	  }
  public void testCloning() {
    EvRealVectorIndividual individual1 = new EvRealVectorIndividual(new double[] {
        0.5d, 0.5d });
    individual1.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
    EvRealVectorIndividual individual2 = individual1.clone();
    assertTrue(individual1.getObjectiveFunctionValue() == individual2.getObjectiveFunctionValue());
    assertTrue( individual1.getValue( 0 ) == individual2.getValue( 0 ));
    individual2.setValue( 0, 3d );
    assertTrue( individual1.getValue( 0 ) != individual2.getValue( 0 ));
    assertTrue(individual1.getObjectiveFunctionValue() != individual2.getObjectiveFunctionValue());
  }
}
