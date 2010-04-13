package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import Gracze.gracz_v3.GeneratorRegulv3;

public class EvStatystykiIleStawiacR1Gatherer extends EvGatherer {

  private int iteration=0;
  
  @Override
  public EvStatistic generate(EvPopulation population) {
    iteration++;
    
    float[] dane = new float[13];
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual) population.getBestResult();
    
    dane[0] = GeneratorRegulv3.regulaIleGracParaWRekuR1.kodGrayaWagi.getWartoscKoduGraya(best);
    dane[1] = GeneratorRegulv3.regulaIleGracParaWRekuR1.kodGrayaWagi.getWartoscKoduGraya(best);
    
    return new EvStatystykaIleStawiacR1Statistic(iteration, null, null);
  }

}
