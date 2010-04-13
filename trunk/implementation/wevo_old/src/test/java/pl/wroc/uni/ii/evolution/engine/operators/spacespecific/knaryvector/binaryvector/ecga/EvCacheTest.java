package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import junit.framework.TestCase;

public class EvCacheTest extends TestCase {

  EvCache cache;
  
  public void testRemove() {
    cache = new EvCache();
    
    EvBlock one = new EvBlock();
    one.put(1);
    EvBlock two = new EvBlock();
    two.put(2);
    
    EvBlock three = new EvBlock();
    three.put(3);
    
    EvBlock result = new EvBlock();
    result.merge(one);
    result.merge(two);
    
 
    cache.put(new EvMergedBlock(one, two, result));    
    cache.put(new EvMergedBlock(new EvBlock(), new EvBlock(), new EvBlock()));
    cache.put(new EvMergedBlock(three, two, result));
    cache.put(new EvMergedBlock(one, three, result));
    
    cache.remove(two, two);
    
    assertEquals(2, cache.getCacheSize());
    
  }


  public void testGetCacheSize() {
    cache = new EvCache();
    assertEquals(0,  cache.getCacheSize());
    
    EvBlock one = new EvBlock();
    one.put(1);
    EvBlock two = new EvBlock();
    two.put(2);
    
    EvBlock result = new EvBlock();
    result.merge(one);
    result.merge(two);
    
    EvMergedBlock mb = new EvMergedBlock(one, two, result);
    cache.put(mb);
    assertEquals(1,  cache.getCacheSize());
    
    assertTrue(cache.getMergedBlock(0) == mb); 
    
  }


}
