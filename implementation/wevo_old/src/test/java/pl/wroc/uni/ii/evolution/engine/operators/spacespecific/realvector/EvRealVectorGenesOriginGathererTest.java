package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import java.util.List;
import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.genesorigin.EvGenesOriginStatistic;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticFileStorage;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.persistency.EvPersistentStatisticStorage;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector.statistic.EvRealVectorGenesOriginGatherer;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;

public class EvRealVectorGenesOriginGathererTest extends TestCase {

  public void testGenerateEvPopulationOfEvRealVectorIndividual() {
    
    
    
    EvPopulation<EvRealVectorIndividual> population = new EvPopulation<EvRealVectorIndividual>();
    
    population.add(new EvRealVectorIndividual(new double[] {1.0, 2.0, 1.0, 1.0, 5.0}));
    population.add(new EvRealVectorIndividual(new double[] {1.0, 2.0, 3.0, 4.0, 5.0}));
    
    population.setObjectiveFunction(new EvRealOneMax<EvRealVectorIndividual>());
    
    
    EvPersistentStatisticStorage storage = new EvPersistentStatisticFileStorage(".");
    EvRealVectorGenesOriginGatherer gatherer = new EvRealVectorGenesOriginGatherer(storage);
    EvGenesOriginStatistic stat = (EvGenesOriginStatistic) gatherer.generate(population);
    
    assertEquals(1.0, stat.genes_discovered[0].get(0));
    assertEquals(1, stat.genes_discovered[0].size());
  
    assertEquals(2.0, stat.genes_discovered[1].get(0));
    assertEquals(1, stat.genes_discovered[1].size());
  
    assertEquals(1.0, stat.genes_discovered[2].get(0));
    assertEquals(3.0, stat.genes_discovered[2].get(1));
    
    
    stat = (EvGenesOriginStatistic) gatherer.generate(population);
    
    for (List<Double> genes: stat.genes_discovered) {
      assertEquals(0, genes.size());
    }
    
   
  
  }

}
