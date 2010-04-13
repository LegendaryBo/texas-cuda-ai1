package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import reguly.kodGraya.KodGraya;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

public class EvStatystykiWymaganychGlosowRXGatherer extends EvGatherer {

 private int iteration = 0;
  
  private KodGraya kodGrayaWymaganych = GeneratorRegul.regulaWymaganychGlosowNaWejscieR1.kodGraya;
  
  private KodGraya kodGrayaPara = GeneratorRegul.regulaCzyParaWReceNaWejscieR1.kodGraya;
  
  private KodGraya kodGrayaKolor = GeneratorRegul.regulaCzyKolorWReceNaWejscieR1.kodGraya;  
  
  private KodGraya kodGrayaOgrStawki = GeneratorRegul.regulaOgraniczenieStawkiNaWejscieR1.kodGrayaWagaGlosu;
  
  private KodGraya kodGrayaOgrStawkiOgraniczenie = GeneratorRegul.regulaOgraniczenieStawkiNaWejscieR1.kodGrayaOgranicznik;
  
  private KodGraya kodGrayaWysokieKarty = GeneratorRegul.regulaWysokieKartyStawkaR1.kodGraya;
  
  private KodGraya kodGrayaBardzoWysokieKarty = GeneratorRegul.regulaBardzoWysokieKartyStawkaR1.kodGraya; 
  
  public EvStatystykiWymaganychGlosowRXGatherer(EvPersistentSimpleStorage storage3) {
    setStorage(storage3);
  }
  


  @Override
  public EvStatistic generate(EvPopulation population) {
    
    float[] dane = new float[7];
    float[] daneBest = new float[7];
    
    
    int[] temp;
    
    for (int i=0; i < population.size(); i++) {
        
      EvBinaryVectorIndividual ind = (EvBinaryVectorIndividual) population.get(i);
      
      dane[0] += kodGrayaWymaganych.getWartoscKoduGraya(ind) / (float)population.size();      
      
      if (ind.getGene(GeneratorRegul.indeksyGenowNaWejscie[0]) == 1)
        dane[1] += kodGrayaPara.getWartoscKoduGraya(ind) / (float)population.size();    
      
      if (ind.getGene(GeneratorRegul.indeksyGenowNaWejscie[1]) == 1)
        dane[2] += kodGrayaKolor.getWartoscKoduGraya(ind) / (float)population.size();     
      
      if (ind.getGene(GeneratorRegul.indeksyGenowNaWejscie[2]) == 1)
        dane[3] += kodGrayaOgrStawki.getWartoscKoduGraya(ind) / (float)population.size();        
    
      if (ind.getGene(GeneratorRegul.indeksyGenowNaWejscie[2]) == 1)
        dane[4] += kodGrayaOgrStawkiOgraniczenie.getWartoscKoduGraya(ind) / (float)population.size();          
      
      if (ind.getGene(GeneratorRegul.indeksyGenowNaWejscie[3]) == 1)
        dane[5] += kodGrayaWysokieKarty.getWartoscKoduGraya(ind) / (float)population.size();        
   
      if (ind.getGene(GeneratorRegul.indeksyGenowNaWejscie[4]) == 1)
        dane[6] += kodGrayaBardzoWysokieKarty.getWartoscKoduGraya(ind) / (float)population.size();             
      
    }
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual) population.getBestResult();
    
    daneBest[0] = kodGrayaWymaganych.getWartoscKoduGraya(best) / (float)population.size();      
    
    if (best.getGene(GeneratorRegul.indeksyGenowNaWejscie[0]) == 1)
      daneBest[1] = kodGrayaPara.getWartoscKoduGraya(best) / (float)population.size();    
    
    if (best.getGene(GeneratorRegul.indeksyGenowNaWejscie[1]) == 1)
      daneBest[2] = kodGrayaKolor.getWartoscKoduGraya(best) / (float)population.size();     
    
    if (best.getGene(GeneratorRegul.indeksyGenowNaWejscie[2]) == 1)
      daneBest[3] = kodGrayaOgrStawki.getWartoscKoduGraya(best) / (float)population.size();        
  
    if (best.getGene(GeneratorRegul.indeksyGenowNaWejscie[2]) == 1)
      daneBest[4] = kodGrayaOgrStawkiOgraniczenie.getWartoscKoduGraya(best) / (float)population.size();          
    
    if (best.getGene(GeneratorRegul.indeksyGenowNaWejscie[3]) == 1)
      daneBest[5] = kodGrayaWysokieKarty.getWartoscKoduGraya(best) / (float)population.size();        
 
    if (best.getGene(GeneratorRegul.indeksyGenowNaWejscie[4]) == 1)
      daneBest[6] = kodGrayaBardzoWysokieKarty.getWartoscKoduGraya(best) / (float)population.size();     
    
    iteration++;
    
    return null;
  }  
    

}
