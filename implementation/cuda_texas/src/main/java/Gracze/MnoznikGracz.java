package Gracze;

public class MnoznikGracz extends Gracz {

  @Override
  public double play(int i, double bid) {
   
    double stawka = (i+1) * 20;
    
    if (gra.stawka > stawka)
      return -1;
    else {
      bilans -= stawka - bid;  
      if (musik > 0) {
        bilans +=musik;
        musik = 0;
      }
    }
    
    return stawka;
  }

}
