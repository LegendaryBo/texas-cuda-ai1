package pl.wroc.uni.ii.evolution.distribution.strategies;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvHyperCube;

public class EvHyperCubeTest extends TestCase {

  public void testGetSet() {
   
    EvHyperCube topology = new EvHyperCube(63);
    long number = 123;
    assertEquals(number, topology.getLongFromBitSet(topology.getBitSetFromLong(number)));
    
    number = Long.MAX_VALUE;
    assertEquals(number, topology.getLongFromBitSet(topology.getBitSetFromLong(number)));    
  }
  
  public void testSome() {
    EvHyperCube topology = new EvHyperCube(2);
    
    assertEquals(4, topology.getCellsCount());
    
  }

}
