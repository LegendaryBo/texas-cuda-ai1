package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

public class EvStatystykaIleStawiacR1Statistic extends EvStatistic {

  public float[] statystykiWagiSrednia;
  public float[] statystykiWagiBest;
  
  public float[] statystykiIle;
  public float[] statystykiIleBest;
  
  private static final long serialVersionUID = -694677918684171417L;

  public EvStatystykaIleStawiacR1Statistic(int iteration, float[] dane, float[] daneBest) {
    
    this.setIteration(iteration);
  
  }    
  
}
