package testy.testyEngine;

import junit.framework.TestCase;
import engine.Karta;
import engine.RegulyGry;
import engine.rezultaty.Rezultat;
import engine.rezultaty.Smiec;

public class TestOkreslaniaRozdan extends TestCase {

  public void testJestStreet() {
    
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 1);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(10, 1);
    karty[5] = new Karta(11, 1);
    karty[6] = new Karta(9, 1);

    assertTrue(RegulyGry.jest_street(karty));
    
    karty[4] = karty[1] = new Karta(7, 2);
    
    assertFalse(RegulyGry.jest_street(karty));
    
    
    karty[0] = new Karta(14, 2);
    karty[1] = new Karta(2, 3);
    karty[2] = new Karta(3, 1);
    karty[3] = new Karta(4, 2);
    karty[4] = new Karta(5, 1);
    karty[5] = new Karta(10, 1);
    karty[6] = new Karta(10, 2);  
    
    assertTrue(RegulyGry.jest_street(karty));
    
    karty[0] = new Karta(5, 3);
    
    assertFalse(RegulyGry.jest_street(karty));
    
    karty[0] = new Karta(14, 3);
    karty[3] = new Karta(11, 2);
    karty[4] = new Karta(12, 1);
    karty[5] = new Karta(13, 1);   
    
    assertTrue(RegulyGry.jest_street(karty));
    
  }
  
  
  public void testJestKolor() {
 
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 1);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(10, 1);
    karty[5] = new Karta(11, 1);
    karty[6] = new Karta(9, 1);    
    
    assertTrue(RegulyGry.jest_kolor(karty));
    
    karty[4] = new Karta(10, 3);
    karty[5] = new Karta(11, 3);  
    
    assertFalse(RegulyGry.jest_kolor(karty));
    
    karty[0] = new Karta(14, 2);
    karty[1] = new Karta(2, 3);
    karty[2] = new Karta(3, 1);
    karty[3] = new Karta(4, 3);
    karty[4] = new Karta(5, 1);
    karty[5] = new Karta(10, 3);
    karty[6] = new Karta(10, 2);     
    
    assertFalse(RegulyGry.jest_kolor(karty));
    
    karty[0] = new Karta(14, 3);
    karty[2] = new Karta(14, 3);
    
    assertTrue(RegulyGry.jest_kolor(karty));
    
  }
  
  public void testPokera() {

    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 1);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 1);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(10, 1);
    karty[5] = new Karta(11, 1);
    karty[6] = new Karta(9, 2);    
    
    assertTrue(RegulyGry.jest_poker(karty));
    
    karty[4] = new Karta(10, 2);
    
    assertFalse(RegulyGry.jest_poker(karty));
    
    
    karty[0] = new Karta(14, 2);
    karty[1] = new Karta(2, 1);
    karty[2] = new Karta(3, 1);
    karty[3] = new Karta(4, 1);
    karty[4] = new Karta(5, 1);
    karty[5] = new Karta(10, 1);
    karty[6] = new Karta(10, 2);  
    
    assertFalse(RegulyGry.jest_poker(karty));
    
    karty[0] = new Karta(14, 1);
    
    assertTrue(RegulyGry.jest_poker(karty)); 
    
  }
  
  public void testIleKartTejSamejWysokosci() {
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 1);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(10, 1);
    karty[5] = new Karta(11, 1);
    karty[6] = new Karta(9, 1);   
    
    byte[] ret = RegulyGry.ile_kart_tej_same_wysokosci(karty);
    assertEquals(ret[0], 2);
    assertEquals(ret[1], 1);
    
    karty[1] = new Karta(10, 1);
    
    ret = RegulyGry.ile_kart_tej_same_wysokosci(karty);
    assertEquals(ret[0], 2);
    assertEquals(ret[1], 2);    

    karty[0] = new Karta(5, 1);
    karty[1] = new Karta(5, 2);
    karty[2] = new Karta(5, 3);
    karty[3] = new Karta(5, 4);
    karty[4] = new Karta(10, 1);
    karty[5] = new Karta(11, 1);
    karty[6] = new Karta(9, 1);     
    
    ret = RegulyGry.ile_kart_tej_same_wysokosci(karty);
    assertEquals(ret[0], 4);
    assertEquals(ret[1], 1);     
    
    karty[0] = new Karta(5, 1);
    karty[1] = new Karta(5, 2);
    karty[2] = new Karta(5, 3);
    karty[3] = new Karta(2, 4);
    karty[4] = new Karta(10, 1);
    karty[5] = new Karta(10, 2);
    karty[6] = new Karta(9, 1); 
    
    ret = RegulyGry.ile_kart_tej_same_wysokosci(karty);
    assertEquals(ret[0], 3);
    assertEquals(ret[1], 2);       
    
  }
  
  
  public void testNiepelnychKart() {
   
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 1);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(0, 0);
    karty[5] = new Karta(0, 0);
    karty[6] = new Karta(0, 0);   
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);    
    System.out.println(rezultat);
    assertTrue(rezultat instanceof Smiec);
    
  }
  
}
