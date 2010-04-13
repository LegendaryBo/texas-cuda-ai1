package wevo.statystyki;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvGatherer;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.EvStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentSimpleStorage;
import reguly.kodGraya.KodGraya;
import Gracze.gracz_v2.generatorRegul.GeneratorRegul;

public class EvStatystykaStawkiRXGatherer extends EvGatherer {

 private int iteration = 0;
  
  private KodGraya kodGrayaPara = null;
  private KodGraya kodGraya2Pary = null;
  private KodGraya kodGrayaTrojka = null;
  private KodGraya kodGrayaStreet = null;
  private KodGraya kodGrayaKolor = null;
  private KodGraya kodGrayaFull = null;
    
  private int runda = 0;

  
  public EvStatystykaStawkiRXGatherer(EvPersistentSimpleStorage storage3, int runda) {
    setStorage(storage3);
    
    this.runda = runda;
    
    kodGrayaPara = new KodGraya(7,
        GeneratorRegul.indeksyGenowStawka1[runda-1]+1 );  
    
    kodGraya2Pary = new KodGraya(8,
        GeneratorRegul.indeksyGenowStawka1[runda-1]+2 + 2);      
    
    kodGrayaTrojka = new KodGraya(9,
        GeneratorRegul.indeksyGenowStawka1[runda-1]+3 + 3 + 2);          
    
    kodGrayaStreet = new KodGraya(10,
        GeneratorRegul.indeksyGenowStawka1[runda-1]+4 + 4 + 3 + 2);      
    
    kodGrayaKolor = new KodGraya(11,
        GeneratorRegul.indeksyGenowStawka1[runda-1]+5 + 5 + 4 + 3 + 2);      
    
    kodGrayaFull = new KodGraya(12,
        GeneratorRegul.indeksyGenowStawka1[runda-1]+6 + 6 + 5 + 4 + 3 + 2);          
    
  }
  


  @Override
  public EvStatistic generate(EvPopulation population) {
    
    float[] dane = new float[6];
    float[] daneBest = new float[6];
    

    for (int i=0; i < population.size(); i++) {
        
        EvBinaryVectorIndividual ind = (EvBinaryVectorIndividual) population.get(i);
      
        if (ind.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1] ) == 1)
          dane[0] += kodGrayaPara.getWartoscKoduGraya(ind) / (float)population.size();
        
        if (ind.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+1 + 2 ) == 1)
          dane[1] += kodGraya2Pary.getWartoscKoduGraya(ind) / (float)population.size();
        
        if (ind.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+2 + 3 + 2 ) == 1)
          dane[2] += kodGrayaTrojka.getWartoscKoduGraya(ind) / (float)population.size();
        
        if (ind.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+3 + 4 + 3 + 2 ) == 1)
          dane[3] += kodGrayaStreet.getWartoscKoduGraya(ind) / (float)population.size();
        
        if (ind.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+4 + 5 + 4 + 3 + 2 ) == 1)
          dane[4] += kodGrayaKolor.getWartoscKoduGraya(ind) / (float)population.size();
        
        if (ind.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+5 + 6 + 5 + 4 + 3 + 2 ) == 1)
          dane[5] += kodGrayaFull.getWartoscKoduGraya(ind) / (float)population.size();
 
    }
    
    EvBinaryVectorIndividual best = (EvBinaryVectorIndividual) population.getBestResult();
    
    if (best.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1] ) == 1)
      daneBest[0] += kodGrayaPara.getWartoscKoduGraya(best);
    
    if (best.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+1 + 2 ) == 1)
      daneBest[1] += kodGraya2Pary.getWartoscKoduGraya(best);
    
    if (best.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+2 + 3 + 2 ) == 1)
      daneBest[2] += kodGrayaTrojka.getWartoscKoduGraya(best);
    
    if (best.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+3 + 4 + 3 + 2 ) == 1)
      daneBest[3] += kodGrayaStreet.getWartoscKoduGraya(best);
    
    if (best.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+4 + 5 + 4 + 3 + 2 ) == 1)
      daneBest[4] += kodGrayaKolor.getWartoscKoduGraya(best);
    
    if (best.getGene( GeneratorRegul.indeksyGenowStawka1[runda-1]+5 + 6 + 5 + 4 + 3 + 2 ) == 1)
      daneBest[5] += kodGrayaFull.getWartoscKoduGraya(best);
    
 
    
    iteration++;
    
    return new EvStatystykaStawkiRXStatistic(iteration, dane, daneBest);
  }
}
