package pl.wroc.uni.ii.evolution.engine.operators.general.combineselector;

import java.util.List;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.combineselector.EvSimpleCombineSelector;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;
import junit.framework.TestCase;

public class EvSimpleCombineSelectorTest extends TestCase {

  public void testReturnEmptyParentsIfPopulationSmallerThanArity() {
  
    EvSimpleCombineSelector<EvStunt> combine_selector = new EvSimpleCombineSelector<EvStunt>();
    combine_selector.init(createPopulation(0), 10, 2);
    assertTrue(combine_selector.getNextParents().size() == 0);
    assertTrue(combine_selector.getNextParents().size() == 0);
  }
  
  
  public void testSimpleResultTest() {
    EvSimpleCombineSelector<EvStunt> combine_selector = new EvSimpleCombineSelector<EvStunt>();
    combine_selector.init(createPopulation(10), 3, 3);
    
    
    /** checks first parents **/
    List<EvStunt> parents = combine_selector.getNextParents();
    assertEquals(3, parents.size());
    
    assertEquals(0.0, parents.get(0).getValue());
    assertEquals(1.0, parents.get(1).getValue());
    assertEquals(2.0, parents.get(2).getValue());

    /** checks second parents **/
    parents = combine_selector.getNextParents();
    assertEquals(3, parents.size());
    
    assertEquals(3.0, parents.get(0).getValue());
    assertEquals(4.0, parents.get(1).getValue());
    assertEquals(5.0, parents.get(2).getValue());
    
    /** checks third parents **/
    parents = combine_selector.getNextParents();
    assertEquals(3, parents.size());
    
    assertEquals(6.0, parents.get(0).getValue());
    assertEquals(7.0, parents.get(1).getValue());
    assertEquals(8.0, parents.get(2).getValue());
    
    
    assertTrue(combine_selector.getNextParents().size() == 0);
    
    List<EvStunt> rest = combine_selector.getUnselectedIndividuals();
    assertEquals(1, rest.size());
    
    assertEquals(9.0, rest.get(0).getValue());
    
  }
  
  
  public void testCommonArityCombineTest() {
    EvSimpleCombineSelector<EvStunt> combine_selector = new EvSimpleCombineSelector<EvStunt>();
    combine_selector.init(createPopulation(11), 2, 5);
    assertTrue(combine_selector.getNextParents().size() == 2);
    assertTrue(combine_selector.getNextParents().size() == 2);
    assertTrue(combine_selector.getNextParents().size() == 2);
    assertTrue(combine_selector.getNextParents().size() == 2);
    assertTrue(combine_selector.getNextParents().size() == 2);
    assertTrue(combine_selector.getNextParents().size() == 0);
    assertTrue(combine_selector.getUnselectedIndividuals().size() == 1);
  }
  
  
  
  public void testManyResultTest() {
    EvSimpleCombineSelector<EvStunt> combine_selector = new EvSimpleCombineSelector<EvStunt>();
    combine_selector.init(createPopulation(5), 3, 5);
 
    /** checks first parents **/
    List<EvStunt> parents = combine_selector.getNextParents();
    assertEquals(3, parents.size());
    
    assertEquals(0.0, parents.get(0).getValue());
    assertEquals(1.0, parents.get(1).getValue());
    assertEquals(2.0, parents.get(2).getValue());
    
    
    /** checks second parents **/
    parents = combine_selector.getNextParents();
    assertEquals(3, parents.size());
    
    assertEquals(1.0, parents.get(0).getValue());
    assertEquals(2.0, parents.get(1).getValue());
    assertEquals(3.0, parents.get(2).getValue());
    
    /** checks third parents **/
    parents = combine_selector.getNextParents();
    assertEquals(3, parents.size());
    
    assertEquals(2.0, parents.get(0).getValue());
    assertEquals(3.0, parents.get(1).getValue());
    assertEquals(4.0, parents.get(2).getValue());
    
    /** checks fourth parents **/
    parents = combine_selector.getNextParents();
    assertEquals(3, parents.size());
    
    assertEquals(3.0, parents.get(0).getValue());
    assertEquals(4.0, parents.get(1).getValue());
    assertEquals(0.0, parents.get(2).getValue());
    
    
  
    assertTrue(combine_selector.getUnselectedIndividuals().size() == 0);
  }
  
  
  
  private EvPopulation<EvStunt> createPopulation(int population_size) {
    
    EvPopulation<EvStunt> pop = new EvPopulation<EvStunt>();
    
    for (int i = 0; i < population_size; i++)  {
      pop.add(new EvStunt(i));
      
    }
    
    return pop;
  }
  
  
}
