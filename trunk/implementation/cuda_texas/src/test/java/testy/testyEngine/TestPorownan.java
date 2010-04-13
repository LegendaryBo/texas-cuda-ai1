package testy.testyEngine;

import junit.framework.TestCase;
import engine.Karta;
import engine.RegulyGry;
import engine.rezultaty.Rezultat;

public class TestPorownan extends TestCase {

  public void testPara() {
    
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(2, 2);
    karty[1] = new Karta(12, 1);
    karty[2] = new Karta(12, 2);
    karty[3] = new Karta(5, 1);
    karty[4] = new Karta(8, 4);
    karty[5] = new Karta(14, 2);
    karty[6] = new Karta(6, 1);    
    
    Karta[] karty2 = new Karta[7];
    karty2[0] = new Karta(2, 2);
    karty2[1] = new Karta(14, 1);
    karty2[2] = new Karta(14, 2);
    karty2[3] = new Karta(3, 1);
    karty2[4] = new Karta(4, 4);
    karty2[5] = new Karta(13, 2);
    karty2[6] = new Karta(6, 1);   
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    Rezultat rezultat2 = RegulyGry.najlepsza_karta(karty2);
    
    assertEquals(rezultat.porownaj(rezultat2), -1);
    assertEquals(rezultat2.porownaj(rezultat), 1);
    
    karty2[1] = new Karta(12, 1);
    karty2[2] = new Karta(12, 2);    
   
    rezultat = RegulyGry.najlepsza_karta(karty);
    rezultat2 = RegulyGry.najlepsza_karta(karty2);    
    
    assertEquals(rezultat.porownaj(rezultat2), 1);
    assertEquals(rezultat2.porownaj(rezultat), -1);
    
    karty[5] = new Karta(13, 2);
    karty2[6] = new Karta(9, 1);  
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    rezultat2 = RegulyGry.najlepsza_karta(karty2);    
    
    assertEquals(rezultat.porownaj(rezultat2), -1);
    assertEquals(rezultat2.porownaj(rezultat), 1);    
    assertEquals(rezultat.porownaj(rezultat), 0); 
     
  }
  
  
  public void testTrojka() {
    
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(2, 2);
    karty[1] = new Karta(12, 1);
    karty[2] = new Karta(12, 2);
    karty[3] = new Karta(5, 1);
    karty[4] = new Karta(8, 4);
    karty[5] = new Karta(12, 3);
    karty[6] = new Karta(6, 1);    
    
    Karta[] karty2 = new Karta[7];
    karty2[0] = new Karta(2, 2);
    karty2[1] = new Karta(2, 1);
    karty2[2] = new Karta(2, 4);
    karty2[3] = new Karta(3, 1);
    karty2[4] = new Karta(4, 4);
    karty2[5] = new Karta(13, 2);
    karty2[6] = new Karta(6, 1); 
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    Rezultat rezultat2 = RegulyGry.najlepsza_karta(karty2); 
    
    assertEquals(rezultat.porownaj(rezultat2), 1);
    assertEquals(rezultat2.porownaj(rezultat), -1);    
    
    karty2[0] = new Karta(12, 2);
    karty2[1] = new Karta(12, 1);
    karty2[2] = new Karta(12, 4);   
    
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    rezultat2 = RegulyGry.najlepsza_karta(karty2); 
    
    assertEquals(rezultat.porownaj(rezultat2), -1);
    assertEquals(rezultat2.porownaj(rezultat), 1);    
    
    karty2[5] = new Karta(8, 2);
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    rezultat2 = RegulyGry.najlepsza_karta(karty2); 
    
    assertEquals(rezultat.porownaj(rezultat2), 0);
    assertEquals(rezultat2.porownaj(rezultat), 0);       
    
    
  }
  
  
  public void testGeneral() {
 
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 1);
    karty[3] = new Karta(9, 1);
    karty[4] = new Karta(9, 3);
    karty[5] = new Karta(11, 1);
    karty[6] = new Karta(9, 4);    
    
    
    Karta[] karty2 = new Karta[7];
    karty2[0] = new Karta(9, 2);
    karty2[1] = new Karta(7, 1);
    karty2[2] = new Karta(4, 1);
    karty2[3] = new Karta(8, 2);
    karty2[4] = new Karta(10, 1);
    karty2[5] = new Karta(11, 1);
    karty2[6] = new Karta(9, 1);   
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    Rezultat rezultat2 = RegulyGry.najlepsza_karta(karty2);     
    
    assertEquals(rezultat.porownaj(rezultat2), 1);
    assertEquals(rezultat2.porownaj(rezultat), -1);   
    
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 1);
    karty[3] = new Karta(2, 2);
    karty[4] = new Karta(9, 3);
    karty[5] = new Karta(11, 1);
    karty[6] = new Karta(14, 4);      
    
    karty2[0] = new Karta(9, 2);
    karty2[1] = new Karta(7, 1);
    karty2[2] = new Karta(3, 1);
    karty2[3] = new Karta(4, 2);
    karty2[4] = new Karta(14, 1);
    karty2[5] = new Karta(11, 3);
    karty2[6] = new Karta(10, 1);  
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    rezultat2 = RegulyGry.najlepsza_karta(karty2);     
    
    assertEquals(rezultat.porownaj(rezultat2), 1);
    assertEquals(rezultat2.porownaj(rezultat), -1);       
    
  }  
  
  
  
  
  
}
