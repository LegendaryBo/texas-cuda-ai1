package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilites;

import java.lang.reflect.Field;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilities.EvRealVectorWithProbabilitesMiLambdaStrategyCrossover;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;

/**
 * @author Lukasz Witko 
 * @author Piotr Baraniak
 */
public class EvMiLambdaStrategiesRecombinationTest extends TestCase {

  
  /**
   * simple tests.
   */
  public void testRecombination() {
    EvPopulation<EvRealVectorWithProbabilitiesIndividual> population = 
         new EvPopulation<EvRealVectorWithProbabilitiesIndividual>();

    EvRealVectorWithProbabilitesMiLambdaStrategyCrossover recombination = 
         new EvRealVectorWithProbabilitesMiLambdaStrategyCrossover();
    
    EvPopulation<EvRealVectorWithProbabilitiesIndividual> new_population;
    
    Class mi_lambda_reflection = recombination.getClass();
    
    EvRealVectorWithProbabilitiesIndividual vector = 
        new EvRealVectorWithProbabilitiesIndividual(
            new double[]{0}, new double[]{0});
    vector.setObjectiveFunction(
        new EvRealOneMax<EvRealVectorWithProbabilitiesIndividual>());
    
    population.add(vector);
    population.add(vector);

    new_population = recombination.apply(population);
    
    assertTrue(population.get(0).getProbability(0) 
        == new_population.get(0).getProbability(0));
    assertTrue(population.get(0).getValue(0) 
        == new_population.get(0).getValue(0));
  
    assertTrue(population.size() == new_population.size());
    
    population.remove(vector);
    
    vector = new EvRealVectorWithProbabilitiesIndividual(
        new double[]{1}, new double[]{0.3});
    vector.setObjectiveFunction(
        new EvRealOneMax<EvRealVectorWithProbabilitiesIndividual>());
    
    population.add(vector);
    new_population = recombination.apply(population);
    
    assertFalse(population.get(0).getProbability(0) 
        == new_population.get(0).getProbability(0));
    
    assertFalse(population.get(0).getValue(0) 
        == new_population.get(0).getValue(0));
    
    assertFalse(population.get(1).getProbability(0) 
        == new_population.get(1).getProbability(0));
    
    assertFalse(population.get(1).getValue(0) 
        == new_population.get(1).getValue(0));
  
    assertTrue(population.size() == new_population.size());
    
    /*This block tests probability in recombination. 
     * It tests if formulas work as in book.*/
    try {
      Field prob_field = mi_lambda_reflection.getDeclaredField("random_value");
      prob_field.setAccessible(true);
      double prob = prob_field.getDouble(recombination);
      double x1_prim = (population.get(0).getValue(0) * prob) 
          + ((1 - prob) * population.get(1).getValue(0));
      double x2_prim = (population.get(1).getValue(0) * prob) 
          + ((1 - prob) * population.get(0).getValue(0));
      double sigma1_prim = (population.get(0).getProbability(0) * prob) 
          + ((1 - prob) * population.get(1).getProbability(0));
      double sigma2_prim = (population.get(1).getProbability(0) * prob) 
          + ((1 - prob) * population.get(0).getProbability(0));
      assertTrue(x1_prim == new_population.get(0).getValue(0));
      assertTrue(x2_prim == new_population.get(1).getValue(0));
      assertTrue(sigma1_prim == new_population.get(0).getProbability(0));
      assertTrue(sigma2_prim == new_population.get(1).getProbability(0));
    } catch (Exception ex) {
      fail(ex.getMessage());
    }

    vector = new EvRealVectorWithProbabilitiesIndividual(
        new double[]{2}, new double[]{0.5});
    
    vector.setObjectiveFunction(
        new EvRealOneMax<EvRealVectorWithProbabilitiesIndividual>());
    
    population.add(vector);

    new_population = recombination.apply(population);
    
    assertFalse(population.get(0).getProbability(0) 
        == new_population.get(0).getProbability(0));
    
    assertFalse(population.get(0).getValue(0) 
        == new_population.get(0).getValue(0));

    assertFalse(population.get(1).getProbability(0) 
        == new_population.get(1).getProbability(0));
    
    assertFalse(population.get(1).getValue(0) 
        == new_population.get(1).getValue(0));
 
    try {
      new_population.get(0).getObjectiveFunctionValue();
      new_population.get(1).getObjectiveFunctionValue();
    } catch (Exception e) {
      fail(e.getMessage());
    }
  }
}
