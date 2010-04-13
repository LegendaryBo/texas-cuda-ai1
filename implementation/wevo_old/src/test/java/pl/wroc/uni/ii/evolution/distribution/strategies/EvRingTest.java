package pl.wroc.uni.ii.evolution.distribution.strategies;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvRing;

public class EvRingTest extends TestCase {

  public void testGetNeighbours() {
    
    /* sanity test */
  
    boolean wyj = false;
    try {
      new EvRing(0);
    } catch (Exception ex) {
      wyj = true;
    }
    assertTrue(wyj);
    
    EvRing top = new EvRing(20);
    
    
    assertEquals(top.getNeighbours(2)[0], 1);
    assertEquals(top.getNeighbours(2)[1], 3);
    assertEquals(2, top.getNeighbours().length);
    
  }

}
