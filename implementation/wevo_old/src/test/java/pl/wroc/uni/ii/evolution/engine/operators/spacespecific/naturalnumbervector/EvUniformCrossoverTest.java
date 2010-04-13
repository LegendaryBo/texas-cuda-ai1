package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.EvNaturalNumberVectorUniformCrossover;

/**
 * 
 * @author Marcin Golebiowski
 */
public class EvUniformCrossoverTest extends TestCase {
  public void testUniformCrossover() throws Exception {

    EvNaturalNumberVectorIndividual indiv1 = new EvNaturalNumberVectorIndividual(new int[] { 2,
        4, 6, 8 });
    EvNaturalNumberVectorIndividual indiv2 = new EvNaturalNumberVectorIndividual(new int[] { 1,
        3, 5, 7 });

    EvPopulation<EvNaturalNumberVectorIndividual> pop = new EvNaturalNumberVectorUniformCrossover().apply(
        new EvPopulation<EvNaturalNumberVectorIndividual>(
            new EvNaturalNumberVectorIndividual[] { indiv1, indiv2 }));
  
    EvNaturalNumberVectorIndividual baby1 = pop.get(0);
    EvNaturalNumberVectorIndividual baby2 = pop.get(1);
    
 
    for (int i = 0; i < 4; i++) {
      int sum = baby1.getNumberAtPosition(i) + baby2.getNumberAtPosition(i) ;
      assertTrue((sum % 2) == 1);
      assertEquals(4 * (i + 1) - 1, sum); 
    }
  
  }

}
