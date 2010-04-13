package testy.testyEngine;

import junit.framework.TestCase;
import Gracze.Gracz;
import Gracze.MinimalistaGracz;
import Gracze.MnoznikGracz;
import Gracze.PassiorGracz;
import Gracze.PodbijaczGracz;
import engine.Gra;
import engine.RegulyGry;

public class TestGry extends TestCase {

  public void testSimple() {
    
    Gracz[] gracze = new Gracz[6];
    gracze[0] = new PassiorGracz();
    gracze[1] = new MinimalistaGracz();
    gracze[2] = new MnoznikGracz();
    gracze[3] = new MinimalistaGracz();
    //gracze[4] = new PassiorGracz();
    gracze[4] = new PodbijaczGracz();
    gracze[5] = new PassiorGracz();
    
    Gra gra = new Gra(gracze);
    
    gra.play_round(false);
    
    for (int i = 0; i < 6; i++) 
      System.out.println(gracze[i] + " " +RegulyGry.najlepsza_karta(gra.rozdanie.getAllCards(i)));
    
    System.out.println(gra.rozdanie);
    
  }
  
}
