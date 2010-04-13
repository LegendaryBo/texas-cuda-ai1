package engine.rezultaty;

import engine.Karta;

public class Poker extends Rezultat {

  public int najwyzsza = 0;
  
  @Override
  public int porownaj_take_same(Rezultat rezultat) {
    Poker przeciwnik = (Poker)rezultat;
    
    if (najwyzsza > przeciwnik.najwyzsza)
      return 1;
    if (najwyzsza < przeciwnik.najwyzsza)
      return -1;        
    
    return 0;
  }      
  
  public Poker(Karta[] karty) {
   
    poziom = 9;
    
    int[] count = new int[5];
    for (int i=0; i < 7; i++) {
      count[karty[i].kolor]++;
    }    
 
    int max_kolor = 0;
    for (int i=1; i <= 4; i++) {
      if (count[i] >= 5)
        max_kolor = i;
    }        
 
    for (int i=0; i < 7; i++) {
      if (karty[i].wysokosc > najwyzsza && karty[i].kolor == max_kolor)
        najwyzsza = karty[i].wysokosc;
    }         
    
  }

}
