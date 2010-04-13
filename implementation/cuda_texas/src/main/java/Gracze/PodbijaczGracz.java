package Gracze;

import java.util.Random;

public class PodbijaczGracz extends Gracz {

  @Override
  public double play(int i, double bid) {
    
    double stawka = 0.0;
     
    
    Random generator = new Random();
    
    if (generator.nextInt(2)==0)
      stawka = gra.stawka*1.5;
    else 
      stawka = gra.stawka;
    bilans -= stawka - bid; 
    if (musik > 0) {
      bilans +=musik;
      musik = 0;
    }
    
    return stawka;
    
  }

}
