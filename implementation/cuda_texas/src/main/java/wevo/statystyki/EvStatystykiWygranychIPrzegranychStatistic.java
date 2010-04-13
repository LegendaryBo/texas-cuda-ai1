package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;

public class EvStatystykiWygranychIPrzegranychStatistic extends EvStatistic {

  public float[] wynikiParti = new float[3];
  public float[] srednieRezultaty = new float[2];
  
  public float[] wynikiPartiBest = new float[3];
  public float[] srednieRezultatyBest = new float[2];  
  
  private static final long serialVersionUID = -694677918684171417L;

  public EvStatystykiWygranychIPrzegranychStatistic(int iteration, float[] dane, float[] daneBest) {
    
    this.setIteration(iteration);
    
    wynikiParti[0] = dane[0];
    wynikiParti[1] = dane[1];
    wynikiParti[2] = dane[2];
    
    srednieRezultaty[0] = dane[3];
    srednieRezultaty[1] = dane[4];
    
    
    wynikiPartiBest[0] = (int)daneBest[0];
    wynikiPartiBest[1] = (int)daneBest[1];
    wynikiPartiBest[2] = (int)daneBest[2];
    
    srednieRezultatyBest[0] = daneBest[3];
    srednieRezultatyBest[1] = daneBest[4];
  }

}
