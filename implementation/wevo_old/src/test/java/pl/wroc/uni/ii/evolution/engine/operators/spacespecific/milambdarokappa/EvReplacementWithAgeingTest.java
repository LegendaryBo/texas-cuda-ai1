package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa.EvMiLambdaRoKappaCheckingAgeComposition;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import junit.framework.TestCase;
/**
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */

public class EvReplacementWithAgeingTest extends TestCase {
  public void testIfItWorks() {
    EvKBestSelection<EvMiLambdaRoKappaIndividual> k_best = new EvKBestSelection<EvMiLambdaRoKappaIndividual>(
        1);
    EvMiLambdaRoKappaCheckingAgeComposition replacement = new EvMiLambdaRoKappaCheckingAgeComposition(1,k_best);
    EvPopulation<EvMiLambdaRoKappaIndividual> population = new EvPopulation<EvMiLambdaRoKappaIndividual>();
    
    EvMiLambdaRoKappaIndividual individual1 = new EvMiLambdaRoKappaIndividual(1);
    individual1.setValue(0, 1d);
    individual1.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());
    individual1.increaseAge();
    
    EvMiLambdaRoKappaIndividual individual2 = new EvMiLambdaRoKappaIndividual(1);
    individual2.setValue(0, 0d);
    individual2.setObjectiveFunction(new EvRealOneMax<EvMiLambdaRoKappaIndividual>());

    population.add(individual1);
    population.add(individual2);
    assertTrue(individual1.getObjectiveFunctionValue() != individual2.getObjectiveFunctionValue());    
    assertEquals(population.size(), 2);
    assertTrue(individual1.getAge() == 1);
    assertTrue(individual2.getAge() == 0);
    
    population = replacement.apply(population);
    
    assertEquals(population.size(), 1);
    
    assertTrue(population.getBestResult().getObjectiveFunctionValue() == individual2.getObjectiveFunctionValue());
    assertTrue(population.getBestResult().getAge() == 0);
  }
}
