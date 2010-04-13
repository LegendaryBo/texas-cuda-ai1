package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilites;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilities.EvRealVectorWithProbabilitiesMiLambdaStrategyMutation;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import junit.framework.TestCase;
/**
 * @author Piotr Baraniak, Tomasz Kozakiewicz, Lukasz Witko
 */
public class EvMiLambdaStrategiesMutationTest extends TestCase {
  public void testIfMutationChangeAnything() {
    
    EvPopulation<EvRealVectorWithProbabilitiesIndividual> population, result_population;
    EvRealVectorWithProbabilitiesMiLambdaStrategyMutation mutation_operator = new EvRealVectorWithProbabilitiesMiLambdaStrategyMutation(0,0);
    
    population = new EvPopulation<EvRealVectorWithProbabilitiesIndividual>();
    EvRealVectorWithProbabilitiesIndividual vector =
      new EvRealVectorWithProbabilitiesIndividual(new double[] {0},new double[] {0});
    vector.setObjectiveFunction( new EvRealOneMax<EvRealVectorWithProbabilitiesIndividual>() );
    population.add(vector);
    try {
      vector.getObjectiveFunctionValue();
    } catch(Exception e) {
      fail(e.getMessage());
    }
    result_population = mutation_operator.apply(population);
    assertTrue(population.get(0).getProbability(0) == result_population.get(0).getProbability(0));
    assertTrue(population.get(0).getValue(0) == result_population.get(0).getValue(0));
    try {
      result_population.get( 0 ).getObjectiveFunctionValue();
    } catch(Exception e) {
      fail(e.getMessage());
    }
    mutation_operator = new EvRealVectorWithProbabilitiesMiLambdaStrategyMutation(1.1,3.1);
    mutation_operator.setMutateClone(true);
    result_population = mutation_operator.apply(population);
    assertTrue(population.get(0).getProbability(0) == result_population.get(0).getProbability(0));
    assertTrue(population.get(0).getValue(0) == result_population.get(0).getValue(0));
    try {
      result_population.get( 0 ).getObjectiveFunctionValue();
    } catch(Exception e) {
      fail(e.getMessage());
    }
    population.get(0).setProbability(0, 2);
    result_population = mutation_operator.apply(population);

    assertFalse(population.get(0).getProbability(0) == result_population.get(0).getProbability(0));
    assertFalse(population.get(0).getValue(0) == result_population.get(0).getValue(0));

    try {
      result_population.get( 0 ).getObjectiveFunctionValue();
    } catch(Exception e) {
      fail(e.getMessage());
    }
  }
}
