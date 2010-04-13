package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;

public class EvStatystykaWygranychOdKartGatherer extends EvGatherer {

private int iteration=0;
  
  public EvStatystykaWygranychOdKartGatherer(EvPersistentSimpleStorage storage3) {
    setStorage(storage3);
  }
  


  @Override
  public EvStatistic generate(EvPopulation population) {
    
    float[] dane = new float[8];
    
    int[] temp;
    
    for (int i=0; i < population.size(); i++) {
      temp = ((EvBinaryVectorIndividual)population.get(i)).wygranaStawka;
      
      for (int j=0; j < 8; j++) 
        dane[j] += temp[j]/(float)population.size();
      
    }
    
    iteration++;
    return new EvStatystykaWygranychOdKartStatistic(iteration, dane);
  }  
    
  
}
