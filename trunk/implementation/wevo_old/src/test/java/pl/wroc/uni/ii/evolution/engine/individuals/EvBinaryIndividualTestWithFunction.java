package pl.wroc.uni.ii.evolution.engine.individuals;

import java.util.ArrayList;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

/**
 * @author Kamil Dworakowski, Jarek Fuks
 * 
 */
public class EvBinaryIndividualTestWithFunction extends TestCase {

  EvBinaryVectorIndividual individual;

  @Override
  protected void setUp() throws Exception {
    individual = new EvBinaryVectorIndividual(2);
    individual.setObjectiveFunction(new EvOneMax());
  }

  public void testAutomaticInvalidate() throws Exception {
    double goal_func_value = individual.getObjectiveFunctionValue();
    individual.setGene(1, 1);
    assertFalse(goal_func_value == individual.getObjectiveFunctionValue());
  }
  
  
  
  public void testString() {
    
    ArrayList<Integer> both = new ArrayList<Integer>();
    both.add(0);
    both.add(1);

    ArrayList<Integer> first = new ArrayList<Integer>();
    first.add(0);
    

    individual.setGene(0, 1);
    individual.setGene(1, 1);
    
    assertEquals("11", individual.toString());
    assertEquals("11", individual.toString(both));
    assertEquals("1", individual.toString(first));
  }
}
