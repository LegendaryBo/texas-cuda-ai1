package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import junit.framework.TestCase;

/**
 * 
 * @author Kacper Gorski
 *
 */
public class EvECGAFastHashMapTest extends TestCase   {

  public void testSimple() {
    EvECGAFastHashMap map = new EvECGAFastHashMap(10);
    
    map.put(4);
    map.put(5);
    
    int[] tab = map.getCount();
    assertEquals(tab.length, 2);
    assertEquals(tab[0], 1);
    assertEquals(tab[0], 1);
    
    map.put(7);
    map.put(5);
    map.put(5);

    tab = map.getCount();
    assertEquals(tab.length, 3);
    assertEquals(tab[0], 1);
    assertEquals(tab[1], 3);    
    assertEquals(tab[2], 1); 
    
  }
  
  public void testConflict() {
    EvECGAFastHashMap map = new EvECGAFastHashMap(10);
    
    map.put(4);
    map.put(14);
    map.put(24);
    map.put(4);
    map.put(24);
    
    int[] tab = map.getCount();
    
    assertEquals(tab.length, 3);
    assertEquals(tab[0], 2);
    assertEquals(tab[1], 1);    
    assertEquals(tab[2], 2);   
    
    map.put(9999);
    map.put(99999);
    map.put(-9999999);
    map.put(999999999);
    map.put(999999999);
    
    tab = map.getCount();
    assertEquals(tab.length, 7);
    
    map.clear();
    tab = map.getCount();
    assertEquals(tab.length, 0);    

    map.put(4);
    map.put(14);
    map.put(24);
    map.put(4);
    map.put(24);
    
    tab = map.getCount();
    
    assertEquals(tab.length, 3);
    assertEquals(tab[0], 2);
    assertEquals(tab[1], 1);    
    assertEquals(tab[2], 2);   
    
    map.put(9999);
    map.put(99999);
    map.put(-9999999);
    map.put(999999999);
    map.put(999999999);
    
    tab = map.getCount();
    assertEquals(tab.length, 7);    
    
  }
  
  
}
