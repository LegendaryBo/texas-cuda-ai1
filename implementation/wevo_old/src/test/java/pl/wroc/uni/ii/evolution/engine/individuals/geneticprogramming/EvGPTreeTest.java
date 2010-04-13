package pl.wroc.uni.ii.evolution.engine.individuals.geneticprogramming;

/**
 * @author Zbigniew Nazimek
 */

//TODO tests !!!

import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPTree;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPType;
import junit.framework.TestCase;

public class EvGPTreeTest extends TestCase {

  public void testGetHeight() {
    EvGPTree t = new EvGPTree(EvGPType.OR,0,3);
    
    assertEquals("Equals 1",1, t.getHeight());
    
    EvGPTree t1 = new EvGPTree(EvGPType.OR,0,3);
    EvGPTree t2 = new EvGPTree(EvGPType.OR,0,3);
    EvGPTree t3 = new EvGPTree(EvGPType.OR,0,3);
    EvGPTree t4 = new EvGPTree(EvGPType.OR,0,3);
    
    t1.setRightSubTree(t2);
    t2.setRightSubTree(t3);
    
    t.setLeftSubTree(t1);
    t.setRightSubTree(t4);
    
    assertEquals("Equals 2",4, t.getHeight());
    
  }

  
  public void testHasRight() {
    EvGPTree t = new EvGPTree(EvGPType.OR,0,3);
    
    t.setRightSubTree(null);    
    assertEquals(false, t.hasRight());
    
    EvGPTree t1 = new EvGPTree(EvGPType.OR,0,3);
    t.setRightSubTree(t1);
    
    assertEquals(true, t.hasRight());
    
  }

  public void testHasLeft() {
    EvGPTree t = new EvGPTree(EvGPType.OR,0,3);
    
    assertEquals(false, t.hasLeft());
    
    EvGPTree t1 = new EvGPTree(EvGPType.OR,0,3);
    t.setLeftSubTree(t1);
    
    assertEquals(true, t.hasLeft());
  }

  public void testSetLeftSubTree() {
    EvGPTree t = new EvGPTree(EvGPType.OR,0,3);
           
    EvGPTree t1 = new EvGPTree(EvGPType.OR,0,3);
    t.setLeftSubTree(t1);
    
    assertTrue(t.getLeftSubTree() == t1);
  }

  public void testSetRightSubTree() {
    EvGPTree t = new EvGPTree(EvGPType.OR,0,3);
    
    EvGPTree t1 = new EvGPTree(EvGPType.OR,0,3);
    t.setRightSubTree(t1);
    
    assertTrue(t.getRightSubTree() == t1);
  }

 
  public void testEval() {
    EvGPTree t = new EvGPTree(EvGPType.OR,0,3);
    EvGPTree t1 = new EvGPTree(EvGPType.CONSTANT,0,3);
    EvGPTree t2 = new EvGPTree(EvGPType.CONSTANT,1.0,3);
    
    t.setLeftSubTree(t1);
    t.setRightSubTree(t2);
    
    assertEquals("equals OR",1.0,t.eval(new double[1]));
    
    t = new EvGPTree(EvGPType.CONSTANT,3.5,3);
    assertEquals("equals CONSTATNT",3.5,t.eval(new double[1]));
    
    t = new EvGPTree(EvGPType.COS,2,4);
    t1 = new EvGPTree(EvGPType.CONSTANT,1.9,5);
    t.setRightSubTree(t1);
    assertEquals("equals COSINUS",Math.cos(1.9),t.eval(new double[1]));
    
    t = new EvGPTree(EvGPType.TABLE_ELEMENT,2,3);
    double[] tab = {0.0,1.0, 2.0, 3.6, 4.1};
    assertEquals("equals TABLE ELEMENT",3.6,t.eval(tab));
    
    
  }

  public void testMutate() {
    EvGPTree t = new EvGPTree(EvGPType.OR,0,3);
    for( int i = 0; i < 100; i++) {
      t.mutate();
      assertNotNull(t);
    }
    
    
  }
  
}
