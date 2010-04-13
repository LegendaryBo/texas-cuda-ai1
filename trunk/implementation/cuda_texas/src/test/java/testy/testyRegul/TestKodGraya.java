package testy.testyRegul;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.kodGraya.KodGraya;

/**
 * Tu jest wyjasnienie: http://en.wikipedia.org/wiki/Gray_code
 * 
 * @author Kacper Gorskir
 *
 */
public class TestKodGraya extends TestCase {

  EvBinaryVectorIndividual osobnikMaly = new EvBinaryVectorIndividual(3);
  EvBinaryVectorIndividual osobnikSredni = new EvBinaryVectorIndividual(8);
  EvBinaryVectorIndividual osobnikDuzy = new EvBinaryVectorIndividual(16);
  
  public void testmaly() {
    
    KodGraya pKodGraya1 = new KodGraya(1, 1);
    KodGraya pKodGraya2 = new KodGraya(1, 2);
    
    KodGraya pKodGraya3 = new KodGraya(2, 1); 
    KodGraya pKodGraya4 = new KodGraya(3, 0); 
    
    osobnikMaly.setGene(1, 1);
    
    assertEquals(1, pKodGraya1.getWartoscKoduGraya(osobnikMaly)); // 1
    assertEquals(0, pKodGraya2.getWartoscKoduGraya(osobnikMaly)); // 0
    
    assertEquals(3, pKodGraya3.getWartoscKoduGraya(osobnikMaly)); // 10
    osobnikMaly.setGene(1, 0);
    assertEquals(0, pKodGraya3.getWartoscKoduGraya(osobnikMaly)); // 00
    
    osobnikMaly.setGene(0, 1); // 100
    assertEquals(7, pKodGraya4.getWartoscKoduGraya(osobnikMaly)); // 100
    osobnikMaly.setGene(2, 1); // 101
    assertEquals(6, pKodGraya4.getWartoscKoduGraya(osobnikMaly)); // 101
    osobnikMaly.setGene(1, 1); // 101
    assertEquals(5, pKodGraya4.getWartoscKoduGraya(osobnikMaly)); // 111
    
  }
  
  
  public void testSredni() {
    KodGraya pKodGraya1 = new KodGraya(4, 1);
    KodGraya pKodGraya2 = new KodGraya(4, 4);
    
    osobnikSredni.setGene(1, 1);
    osobnikSredni.setGene(7, 1);
    osobnikSredni.setGene(6, 1);
    
    assertEquals(15, pKodGraya1.getWartoscKoduGraya(osobnikSredni)); // 1000
    assertEquals(2, pKodGraya2.getWartoscKoduGraya(osobnikSredni)); // 0011
  }
  
  public void testDuzy() {
    for (int i = 0; i < 16; i++)
      osobnikDuzy.setGene(i, 1);
    
    KodGraya pKodGraya1 = new KodGraya(10, 1);
  
  }
  
  
  public void testKodGraya() {
    
    KodGraya pKodGraya1 = new KodGraya(10, 3);
    
    for (int i=0; i < 1023; i++) {
      pKodGraya1.setValue(osobnikDuzy, i);
//      System.out.println(osobnikDuzy);
      assertEquals(i, pKodGraya1.getWartoscKoduGraya(osobnikDuzy));
    }
  }
  
}
