package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;

public class EvStatystykiLicznikPassowGatherer extends EvGatherer {

private int iteration=0;
  
  public EvStatystykiLicznikPassowGatherer(EvPersistentSimpleStorage storage3) {
    setStorage(storage3);
    
  }
  


  @Override
  public EvStatistic generate(EvPopulation population) {
    
    float[] dane = new float[4];
    int[] temp;
    
    for (int i=0; i < population.size(); i++) {
      temp = ((EvBinaryVectorIndividual)population.get(i)).licznikPassow;
      
      for (int j=0; j < 4; j++) 
        dane[j] += temp[j]/population.size();
      
    }
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual)population.getBestResult();
    int[] daneNajlepszego = best.licznikPassow;
    
    
    iteration++;
    return new EvStatystykiLicznikPassowStatistic(iteration, dane, daneNajlepszego);
  }

}
