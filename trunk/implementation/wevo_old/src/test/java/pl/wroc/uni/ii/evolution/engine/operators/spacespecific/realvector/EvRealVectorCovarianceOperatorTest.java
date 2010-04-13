package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvector;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;


/**
 * 
 * @author Szymon Fogiel (szymek.fogiel@gmail.com)
 *
 * Test case for EvRealVectorCovarianceOperator operator.
 */
public class EvRealVectorCovarianceOperatorTest extends TestCase {

  
  /**
   * Tests if given a population with all individuals equal
   * operator return the same population.
   */
  public void testGenerateIdenticalPopulation() {
    EvRealVectorIndividual[] individuals =
      new EvRealVectorIndividual[20];        
    
    EvOperator<EvRealVectorIndividual> operator =
      new EvRealVectorCovarianceOperator(10);
    
    double[] individual_values = new double[] {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
      };
    
    EvRealOneMax<EvRealVectorIndividual> obj_function =
      new EvRealOneMax<EvRealVectorIndividual>();
    
    EvPopulation<EvRealVectorIndividual> population;
    EvPopulation<EvRealVectorIndividual> new_population;
    
    for (int i = 0; i < 20; i++) {
      individuals[i] = new EvRealVectorIndividual(individual_values);
      individuals[i].addObjectiveFunction(obj_function);
    }
    
    population = new EvPopulation<EvRealVectorIndividual>(individuals);
    new_population = operator.apply(population);
    
    EvRealVectorIndividual old_individual = population.get(0);
    EvRealVectorIndividual new_individual = new_population.get(0);
    
    for (int i = 0; i < 10; i++) {
      assertEquals(new_individual.getValue(i), old_individual.getValue(i));
    }
  }
}
