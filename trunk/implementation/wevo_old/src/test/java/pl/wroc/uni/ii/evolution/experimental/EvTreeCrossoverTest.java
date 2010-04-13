package pl.wroc.uni.ii.evolution.experimental;

import org.jmock.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvDecisionTreeIndividual;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvTreeCrossover;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvIsDividedByFive;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvIsDividedByThree;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Kamil Dworakowski
 *
 */
public class EvTreeCrossoverTest extends MockObjectTestCase {
  
  
  //give two trees (single nodes), ensure they will be swapped
  //this depends on the order of the children in resulting population
  public void testCrossoverOnSingleNodesStateTest() throws Exception {
    EvPopulation<EvDecisionTreeIndividual<Integer>> parent_pop =
      new EvPopulation<EvDecisionTreeIndividual<Integer>>();
    
    EvDecisionTreeIndividual<Integer> parent1 = 
      new EvDecisionTreeIndividual<Integer>(new EvIsDividedByFive());
    EvDecisionTreeIndividual<Integer> parent2 = 
      new EvDecisionTreeIndividual<Integer>(new EvIsDividedByThree());
    
    EvTreeCrossover<EvDecisionTreeIndividual<Integer>> operator = 
      new EvTreeCrossover<EvDecisionTreeIndividual<Integer>>(EvRandomizer.INSTANCE);
    
    parent_pop.add(parent1);
    parent_pop.add(parent2);
    
    EvPopulation<EvDecisionTreeIndividual<Integer>> result_pop =
      operator.apply(parent_pop);
    
    assertEquals(2,result_pop.size());
    assertEquals(2,parent_pop.size());
    
    assertEquals(parent2, result_pop.get(0));
    assertEquals(parent1, result_pop.get(1));
    
    // children should be new instances
    assertNotSame(parent2,result_pop.get(0));
    assertNotSame(parent1,result_pop.get(1));
  }
}
