package engine;

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

public class RegulyGry {

  // reka - 7 kart (2 na rece i 5 w srodku)
  public static Rezultat najlepsza_karta(Karta[] karty) {
    
    boolean jest_poker = false;
    
    byte[] karty_tej_same_wysokosci = ile_kart_tej_same_wysokosci(karty);
    
    if (jest_kolor(karty) && jest_street(karty) ) {
      jest_poker = jest_poker(karty);
    }
    
    if (jest_poker) {
      return new Poker(karty);
    }
    if (karty_tej_same_wysokosci[0]==4) {
      return new Kareta(karty);
    }
    if (karty_tej_same_wysokosci[0]==3 && karty_tej_same_wysokosci[1] >= 2) {
      return new Full(karty);
    }    
    if (jest_kolor(karty)) {
      return new Kolor(karty);
    }        
    if (jest_street(karty)) {
      return new Street(karty);
    }       
    if (karty_tej_same_wysokosci[0]==3) {
      return new Trojka(karty);
    }          
    if (karty_tej_same_wysokosci[0]==2 && karty_tej_same_wysokosci[1] == 2) {
      return new DwiePary(karty);
    }     
    if (karty_tej_same_wysokosci[0]==2) {
      return new Para(karty);
    }       
    
    return new Smiec(karty);    
    
  }

  
  
  
  
  
  
  
  
  
  
  public static byte[] ile_kart_tej_same_wysokosci(Karta[] karty) {
    
    byte[] zliczaczka = new byte[15];
    
    for (int i=0; i < 7; i++) {
      zliczaczka[karty[i].wysokosc]++;
    }
  
    
    byte first_place = 0;
    byte second_place = 0;
    for (int i=1; i < 15; i++) {
      if (zliczaczka[i] < first_place && zliczaczka[i] > second_place)
        second_place = zliczaczka[i];
      if (zliczaczka[i] >= first_place) {
        second_place = first_place;        
        first_place = zliczaczka[i];
      }
        
    }
    
    byte[] ret = new byte[2];
    ret[0] = first_place;
    ret[1] = second_place;
    return ret;
  }




  static int i=0;

  public static boolean jest_poker(Karta[] karty) {
    
    byte[][] count = new byte[15][5];
    
    for (int i=0; i < 7; i++) {
      count[karty[i].wysokosc][karty[i].kolor]++;
    }
    count[1][1] = count[14][1];
    count[1][2] = count[14][2];
    count[1][3] = count[14][3];
    count[1][4] = count[14][4];
    
    
    byte dno_pokera = 1;
    byte szczyt_pokera = 1;

    
    for (byte i = 1; i < 15; i++) {
      
      if (count[i][1] > 0)
        szczyt_pokera = i;
      else 
        dno_pokera = (byte) (i + 1);
      
      if (szczyt_pokera - dno_pokera >= 4) {
        return true;
      }
    }    
  
    
    dno_pokera = 1;
    szczyt_pokera = 1;  
    
    for (byte i = 1; i < 15; i++) {
      
      if (count[i][2] > 0)
        szczyt_pokera = i;
      else 
        dno_pokera = (byte) (i + 1);
      
      if (szczyt_pokera - dno_pokera >= 4) {
        return true;
      }
    }        

    dno_pokera = 1;
    szczyt_pokera = 1;  
    
    for (byte i = 1; i < 15; i++) {
      
      if (count[i][3] > 0)
        szczyt_pokera = i;
      else 
        dno_pokera = (byte) (i + 1);
      
      if (szczyt_pokera - dno_pokera >= 4) {
        return true;
      }
    }        
    
    dno_pokera = 1;
    szczyt_pokera = 1;  
    
    for (int i = 1; i < 15; i++) {
      
      if (count[i][4] > 0)
        szczyt_pokera = (byte) i;
      else 
        dno_pokera = (byte) (i + 1);
      
      if (szczyt_pokera - dno_pokera >= 4) {
        return true;
      }
    }        
    return false;
  }




  

  public static boolean jest_kolor(Karta[] karty) {
    byte[] count = new byte[5];
    for (byte i=0; i < 7; i++) {
      count[karty[i].kolor]++;
    }    
    
    if (count[1] >=  5 || count[2] >=  5 || count[3] >=  5 || count[4] >=  5)
      return true;
    
    return false;
  }


  public static boolean jest_street(Karta[] karty) {
    byte[] count = new byte[15];
    
    for (byte i=0; i < 7; i++) {
      count[karty[i].wysokosc]++;
    }
    count[1] = count[14]; // bo as sluzy tez za jedynke
    
    byte dno_streeta = 1;
    byte szczyt_streeta = 1;
    boolean jest_street = false;

    
    for (byte i = 1; i < 15; i++) {
      
      if (count[i] > 0)
        szczyt_streeta = i;
      else 
        dno_streeta = (byte) (i + 1);
      
      if (szczyt_streeta - dno_streeta >= 4) {
        jest_street = true;
      }
    }
    
    return jest_street;
  }
  
  
}
