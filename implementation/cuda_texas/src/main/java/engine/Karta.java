package engine;

/**
 * 
 * Klasa reprezentujaca karte, nie ma tu nic szczegolnego, karta ma dwa pola - wysokosc i kolor
 * 
 * Jest tez metoda toString()
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 *
 */
public class Karta {

  // 2 - dwojka.... 10 - dziesiatka, 11 - walek, 12 dama, 13 krol, 14 as
  public byte wysokosc = 0;
  // 1 - pik, 2 - trefl, 3 - kier, 4 - karo
  public byte kolor = 0;
  
  public Karta(int wysokosc_, int kolor_) {
    wysokosc = (byte) wysokosc_;
    kolor = (byte) kolor_;
  }
  
  
  
  public String toString() {
    String ret = new String();
    
    if (wysokosc == 11)
      ret = "walet";
    if (wysokosc == 12)
      ret = "dama";
    if (wysokosc == 13)
      ret = "krol";
    if (wysokosc == 14)
      ret = "as";        
    if (wysokosc <= 10)
      ret = ""+wysokosc;                
 
    if (kolor==0)
      ret += " blank";        
    if (kolor==1)
      ret += " pik";
    if (kolor==2)
      ret += " trefl";
    if (kolor==3)
      ret += " kier";
    if (kolor==4)
      ret += " karo";   
    
    
    return ret;
    
  }
  
}
