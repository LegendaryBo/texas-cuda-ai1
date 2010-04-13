package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

public class EvStatystykiLicznikPassowStatistic extends EvStatistic{

  public float[] licznikPassow;
  
  public int[] licznikPassowNajlepszego;
  
  private static final long serialVersionUID = -694345L;

  public EvStatystykiLicznikPassowStatistic(int iteration, float[] dane, int[] daneNajlepszego) {
    
    this.setIteration(iteration);

    licznikPassow = dane.clone();
    licznikPassowNajlepszego = daneNajlepszego.clone();
  }  
  
}
