package engine.rezultaty;

import engine.Gra;
import engine.Karta;
import engine.RegulyGry;

/**
 * 
 * Jesli to metoda sluzaca do reprezentacji 'rak'
 * 
 * Jedyna funkcja, ktora powinny implementowac 'rece' jest funkcja do porownywania, 
 *  ktore rozdanie jest lepsze
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 *
 */
public abstract class Rezultat {

  // smiec = 1
  // para = 2
  // 2 pary = 3
  // trojka = 4
  // street = 5
  // kolor = 6
  // full = 7
  // kareta = 8
  // poker = 9
  public int poziom = 0; // im wiekszy poziom, tym lepsza reka
  
  // sluzy do porownyania 2 takich samych rozdan (np dwoch trojek)
  // 1 - aktualne rozdaniw wygrywa, 0 - remis, -1 przegrywa
  abstract int porownaj_take_same(Rezultat rezultat);
  
  
  public int porownaj(Rezultat przeciwnik) {
    if (poziom > przeciwnik.poziom) // banalne przypadki
      return 1;
    if (poziom < przeciwnik.poziom)
      return -1;    
    
    // gdy takie same, to juz zalezy od implementacji obiektow
    return porownaj_take_same(przeciwnik);
    
  }
  
  
  public static Rezultat pobierzPrognoze(Gra aGra, int aKolejnosc) {
    if (aGra.runda == 1 )
      return pobierzPrognoze5(aGra, aKolejnosc);
    if (aGra.runda == 2 )
      return pobierzPrognoze6(aGra, aKolejnosc);
    if (aGra.runda == 3 )
      return pobierzPrognoze7(aGra, aKolejnosc);    
    
    throw new IllegalStateException("niepawidlowa runda "+aGra.runda);
  }
  
  static Karta[] karty = new Karta[7];
  
  private static Rezultat pobierzPrognoze5(Gra aGra, int aKolejnosc) {
    
    karty[0] =  aGra.getPublicCard(0);
    karty[1] =  aGra.getPublicCard(1);
    karty[2] =  aGra.getPublicCard(2);
    karty[3] =  aGra.getPrivateCard(aKolejnosc, 0);
    karty[4] =  aGra.getPrivateCard(aKolejnosc, 1); 
    karty[5] = new Karta(0,0);
    karty[6] = new Karta(0,0);
    
    return RegulyGry.najlepsza_karta(karty);
  }  
  
  
  private static Rezultat pobierzPrognoze6(Gra aGra, int aKolejnosc) {

    karty[0] =  aGra.getPublicCard(0);
    karty[1] =  aGra.getPublicCard(1);
    karty[2] =  aGra.getPublicCard(2);
    karty[3] =  aGra.getPublicCard(3);
    karty[4] =  aGra.getPrivateCard(aKolejnosc, 0); 
    karty[5] =  aGra.getPrivateCard(aKolejnosc, 1); 
    karty[6] = new Karta(0,0);
    
    return RegulyGry.najlepsza_karta(karty);
  }    
  
  public static Rezultat pobierzPrognoze7(Gra aGra, int aKolejnosc) {

    karty[0] =  aGra.getPublicCard(0);
    karty[1] =  aGra.getPublicCard(1);
    karty[2] =  aGra.getPublicCard(2);
    karty[3] =  aGra.getPublicCard(3);
    karty[4] =  aGra.getPublicCard(4);
    karty[5] =  aGra.getPrivateCard(aKolejnosc, 0); 
    karty[6] = aGra.getPrivateCard(aKolejnosc, 1); 
    
    return RegulyGry.najlepsza_karta(karty);
  }    

  
}
