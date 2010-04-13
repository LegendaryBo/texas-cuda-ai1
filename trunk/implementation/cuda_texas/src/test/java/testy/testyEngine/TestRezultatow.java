package testy.testyEngine;

import junit.framework.TestCase;
import engine.Karta;
import engine.RegulyGry;
import engine.rezultaty.DwiePary;
import engine.rezultaty.Full;
import engine.rezultaty.Kareta;
import engine.rezultaty.Kolor;
import engine.rezultaty.Para;
import engine.rezultaty.Poker;
import engine.rezultaty.Rezultat;
import engine.rezultaty.Smiec;
import engine.rezultaty.Street;
import engine.rezultaty.Trojka;

public class TestRezultatow extends TestCase {

  public void testSmiec() {
    
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 3);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(14, 4);
    karty[5] = new Karta(11, 1);
    karty[6] = new Karta(3, 1);
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Smiec);
    
    assertEquals(((Smiec)rezultat).karty[0], 14);
    assertEquals(((Smiec)rezultat).karty[1], 11);
    assertEquals(((Smiec)rezultat).karty[2], 9);
    assertEquals(((Smiec)rezultat).karty[3], 8);
    assertEquals(((Smiec)rezultat).karty[4], 7);
    
    karty[0] = new Karta(13, 2);
    karty[1] = new Karta(2, 3);
    karty[2] = new Karta(3, 1);
    karty[3] = new Karta(4, 3);
    karty[4] = new Karta(5, 1);
    karty[5] = new Karta(10, 3);
    karty[6] = new Karta(7, 2);  

    rezultat = RegulyGry.najlepsza_karta(karty);

    assertTrue(rezultat instanceof Smiec);
    
    assertEquals(((Smiec)rezultat).karty[0], 13);
    assertEquals(((Smiec)rezultat).karty[1], 10);
    assertEquals(((Smiec)rezultat).karty[2], 7);
    assertEquals(((Smiec)rezultat).karty[3], 5);
    assertEquals(((Smiec)rezultat).karty[4], 4);    
    
  }
  
  public void testPara() {
 
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 3);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(14, 4);
    karty[5] = new Karta(3, 2);
    karty[6] = new Karta(3, 1);
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Para);
    
    assertEquals(((Para)rezultat).karty[0], 14);
    assertEquals(((Para)rezultat).karty[1], 9);
    assertEquals(((Para)rezultat).karty[2], 8);
    assertEquals(((Para)rezultat).poziom_pary, 3);
    
    karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 4);
    karty[3] = new Karta(3, 1);
    karty[4] = new Karta(10, 1);
    karty[5] = new Karta(11, 3);
    karty[6] = new Karta(11, 1);
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Para);
    
    assertEquals(((Para)rezultat).karty[0], 10);
    assertEquals(((Para)rezultat).karty[1], 9);
    assertEquals(((Para)rezultat).karty[2], 7);
    assertEquals(((Para)rezultat).poziom_pary, 11);    
    
  }
  
  public void testDwochPar() {
  
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(14, 3);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(14, 4);
    karty[5] = new Karta(3, 2);
    karty[6] = new Karta(3, 1);
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof DwiePary);
    
    assertEquals(((DwiePary)rezultat).wyzsza_para, 14);
    assertEquals(((DwiePary)rezultat).nizsza_para, 3);    
    assertEquals(((DwiePary)rezultat).najwyzsza_karta, 9); 
    
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(4, 4);
    karty[3] = new Karta(10, 1);
    karty[4] = new Karta(10, 1);
    karty[5] = new Karta(11, 3);
    karty[6] = new Karta(11, 1);   
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof DwiePary);
    
    assertEquals(((DwiePary)rezultat).wyzsza_para, 11);
    assertEquals(((DwiePary)rezultat).nizsza_para, 10);    
    assertEquals(((DwiePary)rezultat).najwyzsza_karta, 9); 
    
  }
  
  public void testTrojki() {
    
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(2, 3);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(14, 4);
    karty[5] = new Karta(2, 2);
    karty[6] = new Karta(2, 1);    
 
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Trojka);  
    assertEquals(((Trojka)rezultat).poziom_trojki, 2);
    assertEquals(((Trojka)rezultat).karta1, 14);
    assertEquals(((Trojka)rezultat).karta2, 9);  

    karty[5] = new Karta(14, 2);
    karty[6] = new Karta(14, 1);     
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Trojka);  
    assertEquals(((Trojka)rezultat).poziom_trojki, 14);
    assertEquals(((Trojka)rezultat).karta1, 9);
    assertEquals(((Trojka)rezultat).karta2, 8);  
    
  }
  
  public void testStreeta() {

    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 1);
    karty[2] = new Karta(2, 3);
    karty[3] = new Karta(8, 1);
    karty[4] = new Karta(11, 4);
    karty[5] = new Karta(10, 2);
    karty[6] = new Karta(2, 1);   
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Street);  
    assertEquals(((Street)rezultat).najwyzsza_karta, 11);
    
    karty[0] = new Karta(14, 2);
    karty[1] = new Karta(2, 1);
    karty[2] = new Karta(3, 3);
    karty[3] = new Karta(4, 1);
    karty[4] = new Karta(5, 4);
    karty[5] = new Karta(6, 2);
    karty[6] = new Karta(7, 1);  
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Street);  
    assertEquals(((Street)rezultat).najwyzsza_karta, 7);
   
    
  }
  
  public void testKolor() {
   
    Karta[] karty = new Karta[7];
    karty[0] = new Karta(9, 2);
    karty[1] = new Karta(7, 4);
    karty[2] = new Karta(2, 4);
    karty[3] = new Karta(8, 4);
    karty[4] = new Karta(14, 4);
    karty[5] = new Karta(3, 4);
    karty[6] = new Karta(2, 1);    
 
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Kolor);  
    assertEquals(((Kolor)rezultat).najwyzsza_karta, 14);  
    
    karty[0] = new Karta(9, 3);
    karty[1] = new Karta(7, 4);
    karty[2] = new Karta(2, 4);
    karty[3] = new Karta(8, 4);
    karty[4] = new Karta(14, 4);
    karty[5] = new Karta(9, 4);
    karty[6] = new Karta(7, 3);    
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Kolor);  
    assertEquals(((Kolor)rezultat).najwyzsza_karta, 14);  
    
  }
  
  public void testFull() {
 
    Karta[] karty = new Karta[7];
    
    karty[0] = new Karta(9, 3);
    karty[1] = new Karta(7, 4);
    karty[2] = new Karta(2, 1);
    karty[3] = new Karta(2, 2);
    karty[4] = new Karta(9, 1);
    karty[5] = new Karta(9, 4);
    karty[6] = new Karta(7, 3);   
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Full);  
    assertEquals(((Full)rezultat).trojka, 9);  
    assertEquals(((Full)rezultat).dwojka, 7);  
    
    karty[0] = new Karta(2, 3);
    karty[1] = new Karta(2, 4);
    karty[2] = new Karta(2, 1);
    karty[3] = new Karta(6, 4);
    karty[4] = new Karta(6, 2);
    karty[5] = new Karta(9, 4);
    karty[6] = new Karta(7, 3);   
    
    rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Full);  
    assertEquals(((Full)rezultat).trojka, 2);  
    assertEquals(((Full)rezultat).dwojka, 6); 
    
  }
  
  public void testKarety() {
  
    Karta[] karty = new Karta[7];
    
    karty[0] = new Karta(9, 3);
    karty[1] = new Karta(2, 4);
    karty[2] = new Karta(2, 1);
    karty[3] = new Karta(9, 2);
    karty[4] = new Karta(9, 1);
    karty[5] = new Karta(9, 4);
    karty[6] = new Karta(14, 3);   
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Kareta);  
    assertEquals(((Kareta)rezultat).czworka, 9);  
    assertEquals(((Kareta)rezultat).najwyzsza, 14);      
    
  }
  
  public void testPokera() {
   
    Karta[] karty = new Karta[7];
    
    karty[0] = new Karta(6, 1);
    karty[1] = new Karta(2, 1);
    karty[2] = new Karta(3, 1);
    karty[3] = new Karta(4, 1);
    karty[4] = new Karta(5, 1);
    karty[5] = new Karta(9, 4);
    karty[6] = new Karta(14, 3);   
    
    Rezultat rezultat = RegulyGry.najlepsza_karta(karty);
    
    assertTrue(rezultat instanceof Poker);  
    assertEquals(((Poker)rezultat).najwyzsza, 6);  
   
  }
  

  
  
  
}
