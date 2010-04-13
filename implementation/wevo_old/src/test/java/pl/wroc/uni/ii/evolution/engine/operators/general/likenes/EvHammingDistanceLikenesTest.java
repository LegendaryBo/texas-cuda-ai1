package pl.wroc.uni.ii.evolution.engine.operators.general.likenes;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.likeness.EvHammingDistanceLikenes;

/**
 * 
 * Simple test to check if it works.
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvHammingDistanceLikenesTest extends TestCase {

  /**
   * basic test.
   */
  public void testLikenes() {
    
    EvHammingDistanceLikenes<EvKnaryIndividual> likenes = 
        new EvHammingDistanceLikenes<EvKnaryIndividual>();
    
    EvPopulation<EvKnaryIndividual> population = 
        new EvPopulation<EvKnaryIndividual>();
    
    EvKnaryIndividual pattern = 
        new EvKnaryIndividual(new int[]{0, 0, 0, 0, 0}, 4);    
    
    EvKnaryIndividual individual1 = 
        new EvKnaryIndividual(new int[]{0, 1, 2, 3, 4}, 4);
    EvKnaryIndividual individual2 = 
        new EvKnaryIndividual(new int[]{0, 1, 1, 1, 0}, 4);
    EvKnaryIndividual individual3 = 
        new EvKnaryIndividual(new int[]{1, 1, 1, 1, 1}, 4);  
    EvKnaryIndividual individual4 = 
        new EvKnaryIndividual(new int[]{0, 0, 4, 4, 0}, 4);    

    population.add(individual1);
    population.add(individual2);
    population.add(individual3);
    population.add(individual4);
    
    EvKnaryIndividual result = likenes.getSimilar(population, pattern);
    
    assertEquals(result, individual4);
    
  }

}
