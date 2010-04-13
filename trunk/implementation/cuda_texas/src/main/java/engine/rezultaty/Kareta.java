package engine.rezultaty;

import engine.Karta;

public class Kareta extends Rezultat {

  public int czworka = 0;
  public int najwyzsza = 0;
  
  @Override
  public int porownaj_take_same(Rezultat rezultat) {
    Kareta przeciwnik = (Kareta)rezultat;
    
    if (czworka > przeciwnik.czworka)
      return 1;
    if (czworka < przeciwnik.czworka)
      return -1;      
    if (najwyzsza > przeciwnik.najwyzsza)
      return 1;
    if (najwyzsza< przeciwnik.najwyzsza)
      return -1;         
    
    return 0;
  }        
  
  
  public Kareta(Karta[] karta) {
    
    poziom = 8;
    
    int[] count = new int[15];
    for (int i=0; i < 7; i++) {
      count[karta[i].wysokosc]++;
    }
    
    
    for (int i=14; i >= 2; i--) {
      
      if (count[i] == 4)
        czworka = i;
      
      if (count[i] > 0 && count[i] < 4 && i > najwyzsza ) {
        najwyzsza = i;
      }
      
    }
  }

}
