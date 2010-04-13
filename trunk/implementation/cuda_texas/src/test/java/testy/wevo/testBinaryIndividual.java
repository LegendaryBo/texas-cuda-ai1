package testy.wevo;

import junit.framework.TestCase;
import wevo.EvBinaryVectorIndividual2;

public class testBinaryIndividual extends TestCase {

  public void testIndividual() {
    EvBinaryVectorIndividual2 bla = new EvBinaryVectorIndividual2(13);
    bla.setGene(4, 1);
    for (int i=0; i < 13; i++) {
      if (i==4)
        assertEquals(1,bla.getGene(i));
      else
        assertEquals(0,bla.getGene(i));
    }
    
    bla.setGene(9, 1);
    assertEquals(1,bla.getGene(9));
    bla.setGene(9, 0);
    assertEquals(0,bla.getGene(9));
    
  }
  
  public void bigTest() {
    EvBinaryVectorIndividual2 bla = new EvBinaryVectorIndividual2(1242);
    
    bla.setGene(4, 1);
    for (int i=0; i < 1242; i++) {    
      if (i%2 == 0)
        bla.setGene(i, 1);
    }
    
    for (int i=0; i < 1242; i++) {
      if (i%2 == 0)
        assertEquals(1,bla.getGene(i));
      else
        assertEquals(0,bla.getGene(i));
    }    
    
  }
  
}
