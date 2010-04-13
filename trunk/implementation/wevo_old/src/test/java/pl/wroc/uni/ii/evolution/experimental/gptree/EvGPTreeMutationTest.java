package pl.wroc.uni.ii.evolution.experimental.gptree;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.EvGPTreeMutation;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPTree;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPType;

/**
 * 
 * @author Zbigniew Nazimek
 *
 */
public class EvGPTreeMutationTest extends TestCase {

  
  public void testApply() {
    EvGPTreeMutation tree = new EvGPTreeMutation(0.01);
    EvPopulation<EvGPTree> pop = new EvPopulation<EvGPTree>();
    EvPopulation<EvGPTree> pop1 = tree.apply(pop);
    
    assertTrue("Check if not null", pop1 != null);
    
    pop.add(new EvGPTree(EvGPType.CONSTANT,1,1));
    pop1 = tree.apply(pop);
    
    assertEquals("returned size",1, pop1.size());
    
    pop.add(new EvGPTree(EvGPType.CONSTANT,1,1));
    pop1 = tree.apply(pop);
    assertEquals("returned size",2, pop1.size());
    
    pop.add(new EvGPTree(EvGPType.CONSTANT,1,1));
    pop.add(new EvGPTree(EvGPType.CONSTANT,1,1));
    pop1 = tree.apply(pop);
    assertEquals("returned size",4, pop1.size());
  }

}
