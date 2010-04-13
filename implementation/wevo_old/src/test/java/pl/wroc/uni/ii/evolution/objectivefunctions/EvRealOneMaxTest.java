package pl.wroc.uni.ii.evolution.objectivefunctions;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;

/**
 * @author Kamil Dworakowski, Jarek Fuks
 * 
 */
public class EvRealOneMaxTest extends TestCase {

  EvObjectiveFunction<EvRealVectorIndividual> function;

  @Override
  protected void setUp() throws Exception {

    function = new EvRealOneMax<EvRealVectorIndividual>();
  }

  public void testBestIndividual() throws Exception {

    EvRealVectorIndividual optimal = new EvRealVectorIndividual(2);
    optimal.setValue(0, 1.0d);
    optimal.setValue(1, 1.0d);

    assertEquals(0d, function.evaluate(optimal), 0.0000000001d);
  }

  public void testMedicoreIndividual() {
    EvRealVectorIndividual individual = new EvRealVectorIndividual(2);
    individual.setValue(0, 0d);
    individual.setValue(1, 0d);

    assertEquals(-2d, function.evaluate(individual), 0.00000000000001d);
  }

}
