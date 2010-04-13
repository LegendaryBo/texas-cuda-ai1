package testy.testyRegul;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.ileGrac.IleGracStawkaRX;

public class TestIleGracStawka extends TestCase {

  /**
   * 
   */
  public void testReguly() {
    
    final int GENES  = 50;
    
    EvBinaryVectorIndividual binaryIndividual = new EvBinaryVectorIndividual(GENES);
    for (int i=0; i < GENES; i++) {
      binaryIndividual.setGene(i, 0);
    }
    
    IleGracStawkaRX regula = new IleGracStawkaRX(2, 5, 8);
    
    assertEquals(8+5+5+1, regula.getDlugoscReguly());
    assertEquals(5, regula.getDlugoscWagi());
    assertEquals(5, regula.gray_fi.getDlugoscKodu());
    assertEquals(2+5+1, regula.gray_fi.getPozycjaStartowa());
    assertEquals(5, regula.kodGrayaWagi.getDlugoscKodu());
    assertEquals(2+1, regula.kodGrayaWagi.getPozycjaStartowa());
    
    assertEquals(8, regula.gray_stawka.getDlugoscKodu());
    assertEquals(5+8, regula.gray_stawka.getPozycjaStartowa());
    

  }
  
}
