package Gracze;

public class MinimalistaGracz extends Gracz {

  @Override
  public double play(int i, double bid) { 
    double stawka = gra.stawka;
    
    bilans -= stawka - bid;
    if (musik > 0) {
      bilans += musik;
      musik = 0;
    }
    
    return stawka;
  }

}
