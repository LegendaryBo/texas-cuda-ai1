package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa.EvMiLambdaRoKappaLocalIntermediaryRecombination;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import junit.framework.TestCase;
import java.lang.reflect.Field;
/**
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvLocalIntermediaryRecombinationTest extends TestCase {
  public void testIfItWork() {
    EvPopulation<EvMiLambdaRoKappaIndividual> population 
        = new EvPopulation<EvMiLambdaRoKappaIndividual>();
    EvMiLambdaRoKappaIndividual individual = new EvMiLambdaRoKappaIndividual(
        new double[] { 3d }, new double[] { 3d }, new double[] { 3d });
    individual.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    population.add(individual);
    individual = new EvMiLambdaRoKappaIndividual(new double[] { 2d },
        new double[] { 4d }, new double[] { 6d });
    individual.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    population.add(individual);
    individual = new EvMiLambdaRoKappaIndividual(new double[] { 1d },
        new double[] { 2d }, new double[] { 9d });
    individual.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    population.add(individual);
    EvMiLambdaRoKappaLocalIntermediaryRecombination recombination 
        = new EvMiLambdaRoKappaLocalIntermediaryRecombination();
    EvPopulation<EvMiLambdaRoKappaIndividual> child = recombination
        .apply(population);
    assertTrue(child.size() == 1);
    
    Class local_intermediary = recombination.getClass();
    try {
      Field field = local_intermediary.getDeclaredField("random_u");
      field.setAccessible(true);
      double u = field.getDouble( recombination );
      field = local_intermediary.getDeclaredField("random_k1");
      field.setAccessible(true);
      int k1 = field.getInt( recombination );
      field = local_intermediary.getDeclaredField("random_k2");
      field.setAccessible(true);
      int k2 = field.getInt( recombination );
      assertTrue(k1 != k2);
      double alpha = population.get(k1).getAlpha(0) * u + (1 - u) * population.get(k2).getAlpha(0);
      double sigma = population.get(k1).getProbability(0) * u + (1 - u) * population.get(k2).getProbability(0);
      double x = population.get(k1).getValue(0) * u + (1 - u) * population.get(k2).getValue(0);
      double eps = 0d;
      assertTrue(child.get(0).getAlpha(0) >= alpha - eps && child.get(0).getAlpha(0) <= alpha + eps);
      assertTrue(child.get(0).getProbability(0) >= sigma - eps && child.get(0).getProbability(0) <= sigma + eps);
      assertTrue(child.get(0).getValue(0) >= x - eps && child.get(0).getValue(0) <= x + eps);
    } catch (Exception e) {
      fail(e.getMessage());
    } 
    
    try {
      child.get(0).getObjectiveFunctionValue();
    } catch (Exception e) {
      fail(e.getMessage());
    }
  }
}
