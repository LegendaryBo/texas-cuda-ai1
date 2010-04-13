package pl.wroc.uni.ii.evolution.distribution.strategies;

import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvFullConnected;
import junit.framework.TestCase;

public class EvFullConnectedTest extends TestCase {

  
  public void testCorrectness() {
    
    for (int it = 0; it < 10; it++) {
      EvFullConnected topology = new EvFullConnected(10);
      
      // exptected set
      Set<Long> set1 = new TreeSet<Long>(Arrays.asList(new Long[] {0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L}));
      set1.remove(topology.assignCellID());

      
      // given set
      long[] result = topology.getNeighbours();
      Long[] res2 = new Long[result.length];
      for (int i = 0; i < result.length; i++) {
        res2[i] = result[i];
      }
      Set<Long> set2 = new TreeSet<Long>(Arrays.asList(res2));
      
      // next invoke
      result = topology.getNeighbours();
      res2 = new Long[result.length];
      for (int i = 0; i < result.length; i++) {
        res2[i] = result[i];
      }
      Set<Long> set3 = new TreeSet<Long>(Arrays.asList(res2));
      
      // compere result of two invoke
      
      assertTrue(set3.equals(set2));
      
      // compare two sets
      assertEquals(set1, set2);
    }
    
  }
}
