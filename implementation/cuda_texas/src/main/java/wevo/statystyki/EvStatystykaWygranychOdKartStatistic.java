package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

public class EvStatystykaWygranychOdKartStatistic extends EvStatistic {

  public float[] statystykiKart = new float[7];
  
  private static final long serialVersionUID = -694677918684171417L;

  public EvStatystykaWygranychOdKartStatistic(int iteration, float[] dane) {
    
    this.setIteration(iteration);
    statystykiKart = dane;
  }
  
  
}
