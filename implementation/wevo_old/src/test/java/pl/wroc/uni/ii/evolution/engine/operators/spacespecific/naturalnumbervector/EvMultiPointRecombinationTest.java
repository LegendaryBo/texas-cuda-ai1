package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;


import java.util.List;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.EvNaturalNumberVectorMultiPointCrossover;

import pl.wroc.uni.ii.evolution.objectivefunctions.naturalnumbervector.EvNaturalPattern;

public class EvMultiPointRecombinationTest extends TestCase {
  
  public void testCopyingObjectiveFunction() throws Exception {
    EvNaturalNumberVectorMultiPointCrossover operator = new EvNaturalNumberVectorMultiPointCrossover(2);
    EvPopulation<EvNaturalNumberVectorIndividual> population = 
      new EvPopulation<EvNaturalNumberVectorIndividual>(new EvNaturalNumberVectorIndividual[] { 
          new EvNaturalNumberVectorIndividual(3), new EvNaturalNumberVectorIndividual(3)
      });
    
    EvNaturalPattern function = new EvNaturalPattern(new int[] {1,1,1});
    
    population.setObjectiveFunction(function);
    
    List<EvNaturalNumberVectorIndividual> resulting_population =
      operator.combine(population);
    
    assertEquals(function,resulting_population.get(0).getObjectiveFunction());
    assertEquals(function,resulting_population.get(1).getObjectiveFunction());
  }
  
  public void testOneSwapPoint() throws Exception {
    EvNaturalNumberVectorMultiPointCrossover operator = new EvNaturalNumberVectorMultiPointCrossover(2);
    EvPopulation<EvNaturalNumberVectorIndividual> population = 
      new EvPopulation<EvNaturalNumberVectorIndividual>(new EvNaturalNumberVectorIndividual[] { 
          new EvNaturalNumberVectorIndividual(new int[] {0,0,0}), new EvNaturalNumberVectorIndividual(new int[] {1,1,1})
      });
    
    List<EvNaturalNumberVectorIndividual> resulting_population =
      operator.combine(population);
    
    EvNaturalNumberVectorIndividual child1 = resulting_population.get(0);
    
    assertTrue( child1.hasValue(0) && child1.hasValue(1));
    
  }

}
