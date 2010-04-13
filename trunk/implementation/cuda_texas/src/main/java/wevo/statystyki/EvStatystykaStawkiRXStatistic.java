package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

public class EvStatystykaStawkiRXStatistic extends EvStatistic {

  public float[] statystykiStawki;
  public float[] statystykiStawkiBest;
  
  private static final long serialVersionUID = -694677918684171417L;

  public EvStatystykaStawkiRXStatistic (int iteration, float[] dane, float[] daneBest) {
    
    this.setIteration(iteration);
    statystykiStawki = dane;
    statystykiStawkiBest = daneBest;
  }    
  
}
