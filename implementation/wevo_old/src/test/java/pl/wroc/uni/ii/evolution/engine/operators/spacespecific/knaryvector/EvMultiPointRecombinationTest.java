package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorMultiPointCrossover;


import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

public class EvMultiPointRecombinationTest extends TestCase {
  
  public void testCopyingObjectiveFunction() throws Exception {
    EvKnaryVectorMultiPointCrossover<EvBinaryVectorIndividual> operator = 
      new EvKnaryVectorMultiPointCrossover<EvBinaryVectorIndividual>(2);
    EvPopulation<EvBinaryVectorIndividual> population = 
      new EvPopulation<EvBinaryVectorIndividual>(new EvBinaryVectorIndividual[] { 
          new EvBinaryVectorIndividual(3), new EvBinaryVectorIndividual(3)
      });
    
    EvOneMax function = new EvOneMax();
    
    population.setObjectiveFunction(function);
    

    EvPopulation<EvBinaryVectorIndividual> resulting_population =
      operator.apply(population);

    
    assertEquals(function,resulting_population.get(0).getObjectiveFunction());
    assertEquals(function,resulting_population.get(1).getObjectiveFunction());

  }
  
  public void testOneSwapPoint() throws Exception {
    EvKnaryVectorMultiPointCrossover<EvBinaryVectorIndividual> operator = 
      new EvKnaryVectorMultiPointCrossover<EvBinaryVectorIndividual>(2);
    EvPopulation<EvBinaryVectorIndividual> population = 
      new EvPopulation<EvBinaryVectorIndividual>(new EvBinaryVectorIndividual[] { 
          new EvBinaryVectorIndividual(new int[] {0 , 0, 0}), new EvBinaryVectorIndividual(new int[] {1, 1, 1})
      });
    population.setObjectiveFunction(new EvOneMax());
    
    EvPopulation<EvBinaryVectorIndividual> resulting_population =
      operator.apply(population);
    
    EvBinaryVectorIndividual child1 = resulting_population.get(0);
    
    assertTrue( child1.getObjectiveFunctionValue() > 0.5 && child1.getObjectiveFunctionValue() < 2.5);
    
  }

}
