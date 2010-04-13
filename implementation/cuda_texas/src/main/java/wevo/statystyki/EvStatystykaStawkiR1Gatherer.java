package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import reguly.kodGraya.KodGraya;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

public class EvStatystykaStawkiR1Gatherer extends EvGatherer {

  private int iteration = 0;
  
  private KodGraya kodGrayaPara = new KodGraya(GeneratorRegul.stawkaR1ParaWReceDlugosc,
      GeneratorRegul.indeksyGenowStawka1[0]+1);  
  
  private KodGraya kodGrayaKolor1 = new KodGraya(GeneratorRegul.stawkaR1KolorWReceDlugosc,
      GeneratorRegul.indeksyGenowStawka1[1]+1);  
  private KodGraya kodGrayaKolor2 = new KodGraya(GeneratorRegul.stawkaR1KolorWReceDlugosc,
      GeneratorRegul.indeksyGenowStawka1[2]+1);  
  
  private KodGraya kodGrayaWysokaKarta = new KodGraya(GeneratorRegul.stawkaR1WysokaKartaWReceDlugosc,
      GeneratorRegul.indeksyGenowStawka1[3]+1);    
  private KodGraya kodGrayaBardzoWysokaKarta = new KodGraya(GeneratorRegul.stawkaR1WysokaKartaWReceDlugosc,
      GeneratorRegul.indeksyGenowStawka1[4]+1);      
  
  private KodGraya kodGrayaStalaStawka3 = new KodGraya(3,
      GeneratorRegul.indeksyGenowStawka1[5]+1);    
  
  private KodGraya kodGrayaStalaStawka5 = new KodGraya(5,
      GeneratorRegul.indeksyGenowStawka1[6]+1);    
  
  private KodGraya kodGrayaStalaStawka10 = new KodGraya(10,
      GeneratorRegul.indeksyGenowStawka1[7]+1);    
  
  public EvStatystykaStawkiR1Gatherer(EvPersistentSimpleStorage storage3) {
    setStorage(storage3);
  }
  


  @Override
  public EvStatistic generate(EvPopulation population) {
    
    float[] dane = new float[8];
    float[] daneBest = new float[8];
    
    
    int[] temp;
    
    for (int i=0; i < population.size(); i++) {
        
        EvBinaryVectorIndividual ind = (EvBinaryVectorIndividual) population.get(i);
      
        if (ind.getGene(GeneratorRegul.indeksyGenowStawka1[0]) == 1)
          dane[0] += kodGrayaPara.getWartoscKoduGraya(ind) / (float)population.size();
        
        if (ind.getGene(GeneratorRegul.indeksyGenowStawka1[1]) == 1)
          dane[1] += kodGrayaKolor1.getWartoscKoduGraya(ind) / (float)population.size();    
        
        if (ind.getGene(GeneratorRegul.indeksyGenowStawka1[2]) == 1)
          dane[2] += kodGrayaKolor2.getWartoscKoduGraya(ind) / (float)population.size();    
        
        if (ind.getGene(GeneratorRegul.indeksyGenowStawka1[3]) == 1)
          dane[3] += kodGrayaWysokaKarta.getWartoscKoduGraya(ind) / (float)population.size(); 
        
        if (ind.getGene(GeneratorRegul.indeksyGenowStawka1[4]) == 1)
          dane[4] += kodGrayaBardzoWysokaKarta.getWartoscKoduGraya(ind) / (float)population.size();   
        
        if (ind.getGene(GeneratorRegul.indeksyGenowStawka1[5]) == 1)
          dane[5] += kodGrayaStalaStawka3.getWartoscKoduGraya(ind) / (float)population.size();   
        
        if (ind.getGene(GeneratorRegul.indeksyGenowStawka1[6]) == 1)
          dane[6] += kodGrayaStalaStawka5.getWartoscKoduGraya(ind) / (float)population.size();   
        
        if (ind.getGene(GeneratorRegul.indeksyGenowStawka1[7]) == 1)
          dane[7] += kodGrayaStalaStawka10.getWartoscKoduGraya(ind) / (float)population.size();   
 
    }
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual) population.getBestResult();
    
    if (best.getGene(GeneratorRegul.indeksyGenowStawka1[0]) == 1)
      daneBest[0] = kodGrayaPara.getWartoscKoduGraya(best);
    
    if (best.getGene(GeneratorRegul.indeksyGenowStawka1[1]) == 1)
      daneBest[1] = kodGrayaKolor1.getWartoscKoduGraya(best);    
    
    if (best.getGene(GeneratorRegul.indeksyGenowStawka1[2]) == 1)
      daneBest[2] = kodGrayaKolor2.getWartoscKoduGraya(best);    
    
    if (best.getGene(GeneratorRegul.indeksyGenowStawka1[3]) == 1)
      daneBest[3] = kodGrayaWysokaKarta.getWartoscKoduGraya(best); 
    
    if (best.getGene(GeneratorRegul.indeksyGenowStawka1[4]) == 1)
      daneBest[4] = kodGrayaBardzoWysokaKarta.getWartoscKoduGraya(best);   
    
    if (best.getGene(GeneratorRegul.indeksyGenowStawka1[5]) == 1)
      daneBest[5] = kodGrayaStalaStawka3.getWartoscKoduGraya(best);   
    
    if (best.getGene(GeneratorRegul.indeksyGenowStawka1[6]) == 1)
      daneBest[6] = kodGrayaStalaStawka5.getWartoscKoduGraya(best);   
    
    if (best.getGene(GeneratorRegul.indeksyGenowStawka1[7]) == 1)
      daneBest[7] = kodGrayaStalaStawka10.getWartoscKoduGraya(best);   
    
    iteration++;
    
    return new EvStatystykaStawkiR1Statistic(iteration, dane, daneBest);
  }  
}
