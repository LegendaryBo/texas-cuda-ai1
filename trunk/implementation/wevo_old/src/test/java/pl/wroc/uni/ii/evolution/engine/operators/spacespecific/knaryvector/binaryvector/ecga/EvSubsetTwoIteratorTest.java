package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import junit.framework.TestCase;

public class EvSubsetTwoIteratorTest extends TestCase {
  EvECGAStructure struct;
  EvSubsetTwoIterator iterator;
  
  public void testHasNext() {
    struct = new EvECGAStructure();
    iterator = new EvSubsetTwoIterator(struct.getSetCount());
    assertFalse(iterator.hasNext());
    
    struct = new EvECGAStructure();
    struct.addBlock(new EvBlock());
    struct.addBlock(new EvBlock());
    iterator = new EvSubsetTwoIterator(struct.getSetCount());
    assertTrue(iterator.hasNext());
  }

  public void testNext() {
    struct = new EvECGAStructure();
    
    struct.addBlock(new EvBlock());
    struct.addBlock(new EvBlock());
    struct.addBlock(new EvBlock());
    struct.addBlock(new EvBlock());
    iterator = new EvSubsetTwoIterator(struct.getSetCount());
    int count = 0;
    while (iterator.hasNext()) {
    
     iterator.next();
      
      count++;
    }
    
    assertEquals(6, count);
  }

}
