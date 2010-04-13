package pl.wroc.uni.ii.evolution.distribution.strategies;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvGrid;

public class EvGridTest extends TestCase {

  private long[] neighbours;

  public void testCorrectness() {
    
    EvGrid topology = new EvGrid(3, 4);
    
    assertEquals(12, topology.getCellsCount());
    
    // test if 0 has (1, 4) neighbours
    neighbours = topology.getNeighbours(0);
    assertEquals(2, neighbours.length);
    assertTrue((neighbours[0] == 1 && neighbours[1] == 4)
        || (neighbours[0] == 4 && neighbours[1] == 1));
    
   
    topology = new EvGrid(2, 2);
    // test if 1 has (0, 3) neighbours
    neighbours = topology.getNeighbours(1);
    assertEquals(2, neighbours.length);
    assertTrue((neighbours[0] == 0 && neighbours[1] == 3)
        || (neighbours[0] == 3 && neighbours[1] == 0));
    
    
  }
}
