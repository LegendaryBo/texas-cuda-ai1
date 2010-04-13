package pl.wroc.uni.ii.evolution.solutionspaces;

import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.EvGPTreeSolutionSpace;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPTree;
import junit.framework.TestCase;
/**
 * 
 * @author Zbigniew Nazimek
 *
 */
public class EvGPTreeSolutionSpaceTest extends TestCase {

  public void testGenerateIndividual() {
    EvGPTreeSolutionSpace s = new EvGPTreeSolutionSpace(null, 10);
    EvGPTree t;
    
    for (int i = 0; i < 20; i++) {
      t = s.generateIndividual();
      assertNotNull(t);
      assertTrue(t.getHeight() <= 10);
    }
    
  }

}
