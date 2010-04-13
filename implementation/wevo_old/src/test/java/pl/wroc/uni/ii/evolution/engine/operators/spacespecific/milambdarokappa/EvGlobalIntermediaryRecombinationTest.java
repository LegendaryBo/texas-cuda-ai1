package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa.EvMiLambdaRoKappaGlobalIntermediaryRecombination;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import junit.framework.TestCase;

public class EvGlobalIntermediaryRecombinationTest extends TestCase {
  public void testIfItWork() {
    
    
    EvPopulation<EvMiLambdaRoKappaIndividual> population = new EvPopulation<EvMiLambdaRoKappaIndividual>();
    
    
    EvMiLambdaRoKappaIndividual individual = new EvMiLambdaRoKappaIndividual(new double[]{3d}, new double[]{3d},new double[] {3d});
    individual.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    population.add(individual);
    
    individual = new EvMiLambdaRoKappaIndividual(new double[]{2d}, new double[]{4d},new double[] {6d});
    individual.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    population.add(individual);
    
    individual = new EvMiLambdaRoKappaIndividual(new double[]{1d}, new double[]{2d},new double[] {9d});
    individual.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    population.add(individual);
    
    EvMiLambdaRoKappaGlobalIntermediaryRecombination recombination = new EvMiLambdaRoKappaGlobalIntermediaryRecombination();
    EvPopulation<EvMiLambdaRoKappaIndividual> child = recombination.apply(population);
    
    assertTrue(child.size() == 1);
    assertTrue(child.get(0).getAlpha(0) == 6d);
    assertEquals(child.get(0).getProbability(0), 3d);
    assertTrue(child.get(0).getValue(0) == 2d);

    try {
      child.get(0).getObjectiveFunctionValue();
    } catch(Exception e) {
      fail(e.getMessage());
    }
  }
}
