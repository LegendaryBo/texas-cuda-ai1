package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

public class EvStatystykaWymaganychGlosowR1Statistic extends EvStatistic {

  public float[] wymaganychGlosow = new float[1];
  public float[] wymaganychGlosowBest = new float[1];
  
  private static final long serialVersionUID = -694677918684171417L;

  public EvStatystykaWymaganychGlosowR1Statistic (int iteration, float[] dane, float[] daneBest) {
    
    this.setIteration(iteration);
    wymaganychGlosow = dane;
    wymaganychGlosowBest = daneBest;
  }
  
  
}
