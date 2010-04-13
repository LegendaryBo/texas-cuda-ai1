package testy.testyRegul;

import junit.framework.TestCase;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

/**
 * wyswietla informacje odnoscnie osobnika
 * 
 */
public class RegulyInformacja extends TestCase {

  public void testInformacje() {
    
    GeneratorRegul.init();
    
    System.out.println("Calkowity rozmiar genomu: "+GeneratorRegul.rozmiarGenomu);
    System.out.println(GeneratorRegul.getInfo());
    
    
    
  }
  
}
