package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;

public class EvStatystykiKartWygranegoGatherer extends EvGatherer{

 private int iteration=0;
  
  public EvStatystykiKartWygranegoGatherer(EvPersistentSimpleStorage storage3) {
    setStorage(storage3);
  }
  


  @Override
  public EvStatistic generate(EvPopulation population) {
    
    float[] dane = new float[7];
    
    int[] temp;
    
    for (int i=0; i < population.size(); i++) {
      temp = ((EvBinaryVectorIndividual)population.get(i)).kartaWygranego;
      
      for (int j=0; j < 7; j++) 
        dane[j] += temp[j]/(float)population.size();
      
    }
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual)population.getBestResult();
    int[] daneNajlepszego = best.kartaWygranego;
    
    iteration++;
    return new EvStatystykiKartWygranegoStatistic(iteration, dane, daneNajlepszego);
  }  
  
  
}
