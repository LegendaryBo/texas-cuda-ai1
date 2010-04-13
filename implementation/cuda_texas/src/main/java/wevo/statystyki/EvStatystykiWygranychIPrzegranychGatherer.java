package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;

/**
 * 
 * Statystyki zbieraja dane o ilosc przegranych/wygranych/spasowanych gier
 * 
 * @author Kacper Gorski 
 *
 */
public class EvStatystykiWygranychIPrzegranychGatherer extends EvGatherer {

  private int iteration=0;
  
  public EvStatystykiWygranychIPrzegranychGatherer(EvPersistentSimpleStorage storage3) {
    setStorage(storage3);
    
  }
  


  @Override
  public EvStatistic generate(EvPopulation population) {
    
    float[] dane = new float[5];
    
    float[] temp;
    
    for (int i=0; i < population.size(); i++) {
      temp = ((EvBinaryVectorIndividual)population.get(i)).statystyki;
      
      for (int j=0; j < 5; j++) 
        dane[j] += temp[j]/population.size();
      
    }
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual)population.getBestResult();
    float[] daneNajlepszego = best.statystyki;
    
    
    iteration++;
    return new EvStatystykiWygranychIPrzegranychStatistic(iteration, dane, daneNajlepszego);
  }

}
