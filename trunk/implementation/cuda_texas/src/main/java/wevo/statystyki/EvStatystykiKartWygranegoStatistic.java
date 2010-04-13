package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

public class EvStatystykiKartWygranegoStatistic extends EvStatistic {

  public float[] statystykiKart = new float[7];
  public int[] statystykiKartBest = new int[7];
  
  private static final long serialVersionUID = -694677918684171417L;

  public EvStatystykiKartWygranegoStatistic(int iteration, float[] dane, int[] daneBest) {
    
    this.setIteration(iteration);
    statystykiKart = dane;
    statystykiKartBest = daneBest;
  }


  
}
