package testy.testyRegul;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.ileGrac.IleGracParaWRekuR1;

public class testIleGracParaWRekuR1 extends TestCase {

  
  /**
   * 
   */
  public void testReguly() {
    
    final int GENES  = 50;
    
    EvBinaryVectorIndividual binaryIndividual = new EvBinaryVectorIndividual(GENES);
    for (int i=0; i < GENES; i++) {
      binaryIndividual.setGene(i, 0);
    }
    
    IleGracParaWRekuR1 regula = new IleGracParaWRekuR1(3, 5);
    
    assertEquals(5+5+1, regula.getDlugoscReguly());
    assertEquals(5, regula.getDlugoscWagi());
    assertEquals(5, regula.gray_fi.getDlugoscKodu());
    assertEquals(4+5, regula.gray_fi.getPozycjaStartowa());
    assertEquals(5, regula.kodGrayaWagi.getDlugoscKodu());
    assertEquals(4, regula.kodGrayaWagi.getPozycjaStartowa());
    

  }
  
}
