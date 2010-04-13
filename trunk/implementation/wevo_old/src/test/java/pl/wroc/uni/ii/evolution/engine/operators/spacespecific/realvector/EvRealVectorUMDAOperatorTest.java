package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Test case for the UMDAc operator.
 * 
 * @author Krzysztof Sroka (krzysztof.sroka@gmail.com)
 */
public class EvRealVectorUMDAOperatorTest extends TestCase {
  
  /**
   * Tests whether UMDA operator generates populations of requested size.
   */
  public void testPopulationSize() {
    int[] sizes = new int[] {1, 10, 1000};
    EvPopulation<EvRealVectorIndividual> population = getIdenticalPopulation();
    
    for (int size : sizes) {
      EvOperator<EvRealVectorIndividual> umdac = 
        new EvRealVectorUMDAOperator(size);
      EvPopulation<EvRealVectorIndividual> new_population =
        umdac.apply(population);
      assertEquals(size, new_population.size());
    }
  }
  
  /**
   * Tests whether UMDA operator creates individuals equal to the individuals
   * from input population if the input population consists of identical 
   * individuals. 
   */
  public void testIdenticalPopulation() {
    EvOperator<EvRealVectorIndividual> umdac = new EvRealVectorUMDAOperator(1);
    EvPopulation<EvRealVectorIndividual> population = getIdenticalPopulation();
    EvPopulation<EvRealVectorIndividual> new_population = 
      umdac.apply(population);
    
    EvRealVectorIndividual old_individual = population.get(0);
    EvRealVectorIndividual new_individual = new_population.get(0);
    
    for (int i = 0; i < 10; i++) {
      assertEquals(new_individual.getValue(i), old_individual.getValue(i));
    }
  }

  /**
   * Helper function generating small population containing identical 
   * individuals.
   *  
   * @return Population with individuals equal to each other.
   */
  private EvPopulation<EvRealVectorIndividual> getIdenticalPopulation() {
    EvRealVectorIndividual[] individuals = new EvRealVectorIndividual[10];
    double[] individual_values = new double[] {
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
    };
    
    for (int i = 0; i < 10; i++) {
      individuals[i] = new EvRealVectorIndividual(individual_values);
    }
    
    return new EvPopulation<EvRealVectorIndividual>(individuals);
  }
}
