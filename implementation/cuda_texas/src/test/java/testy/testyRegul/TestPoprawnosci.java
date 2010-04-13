package testy.testyRegul;

import java.util.ArrayList;

import junit.framework.TestCase;
import reguly.RegulaAbstrakcyjna;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;
import Gracze.gracz_v3.GeneratorRegulv3;

/**
 * 
 * Sprawdza, czy w individualu nie ma regul, ktore uzywaja tego samego genu
 * 
 * @author Kacper Gorski
 *
 */
public class TestPoprawnosci extends TestCase {

  /**
   * tworzymy tablice o rozmiarze takim jak rozmiar osobnika i zliczamy, ktory gen jest przez ile regul uzywany
   *
   */
  public void testPoprawnosc() {
    
    GeneratorRegul.init();
    
    ArrayList<RegulaAbstrakcyjna> pListaWszystkichRegul = GeneratorRegulv3.generujKompletRegul(null);
    
    int[] pTablicaGenow = new int[GeneratorRegulv3.rozmiarGenomu];
    
    for (RegulaAbstrakcyjna pRegula : pListaWszystkichRegul) {
	 
      for (int i=pRegula.getPozycjaStartowaWGenotypie(); 
      		i < pRegula.getPozycjaStartowaWGenotypie() + pRegula.getDlugoscReguly(); 
      		i++) {
        pTablicaGenow[i]++;
        if (pTablicaGenow[i]>1) {
          System.out.println("blad w regule "+pRegula+" i: "+i);
          System.out.println("pozycja startowa w genotypie "+ pRegula.getPozycjaStartowaWGenotypie());
          System.out.println("dlugosc reguly "+ pRegula.getDlugoscReguly());
          
          
          fail();
        }
      }
    }
    
//    for (int i=0; i < GeneratorRegul.rozmiarGenomu; i++) {
//      System.out.println(i + " "+pTablicaGenow[i]);
//    }
    
  }
  
}
