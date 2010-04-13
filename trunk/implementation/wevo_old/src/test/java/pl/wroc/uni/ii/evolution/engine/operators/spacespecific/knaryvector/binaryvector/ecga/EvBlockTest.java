package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import junit.framework.TestCase;

public class EvBlockTest extends TestCase {

  public void testEqualsObject() {
    EvBlock one = new EvBlock();
    EvBlock two = new EvBlock();
    
    assertTrue(one.equals(two));
    assertFalse(one.equals(1));
    
    one.put(1);
    one.put(3);
    
    assertFalse(one.equals(two));
    
    two.put(3);
    two.put(1);
    
    assertTrue(one.equals(two));
    
    two.put(4);
    
    assertFalse(one.equals(two));
  }

}