package pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness;

import java.lang.reflect.Field;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvNearZeroFitness;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
/**
 * @author Piotr Baraniak
 */
public class EvNearZeroFitnessTest extends TestCase {
  @SuppressWarnings("unchecked")
  public void testIfItWorks() {
    EvPopulation<EvMiLambdaRoKappaIndividual> population = new EvPopulation<EvMiLambdaRoKappaIndividual>();
    EvMiLambdaRoKappaIndividual individual = new EvMiLambdaRoKappaIndividual(1);
    individual.setObjectiveFunction( new EvRealOneMax<EvMiLambdaRoKappaIndividual>() );
    individual.setValue( 0, 0.5 );
    population.add( individual );
    individual = new EvMiLambdaRoKappaIndividual(1);
    individual.setObjectiveFunction( new EvRealOneMax<EvMiLambdaRoKappaIndividual>() );
    individual.setValue( 0, -1d );
    population.add( individual );
    double eps = 0.000001;
    
    /* Creating fitness */
    EvNearZeroFitness<EvMiLambdaRoKappaIndividual> fitness = new EvNearZeroFitness<EvMiLambdaRoKappaIndividual>();
    fitness.reinitialize(population);
    double fit = fitness.getFitness(population.get( 0 ));
    assertEquals(1d,fit);
    fitness.reinitialize(population);
    fit = fitness.getFitness(population.get( 1 ));
    assertEquals(0d,fit);
    Class fitness_reflection = fitness.getClass();
    double fitness_sum = 0;
    try {
      Field sum  = fitness_reflection.getDeclaredField("sum");
      sum.setAccessible(true);
      fitness_sum = (double) sum.getDouble( fitness ); 
      assertEquals("Sum has wrong value",1.25,fitness_sum );
    } catch( Exception ex) {
      fail(ex.getMessage());
    }
    double sum_of_fitnesses = 0;
    for( int i = 0; i < population.size(); i++) {
      fitness.reinitialize(population);
      sum_of_fitnesses += fitness.getFitness( population.get( i ));
    }
    assertTrue("Sum of fitnesses isn't 1.", sum_of_fitnesses >= 1 - eps && sum_of_fitnesses <= 1 + eps);
    /* Adding next individual*/
    individual = new EvMiLambdaRoKappaIndividual(1);
    individual.setObjectiveFunction( new EvRealOneMax<EvMiLambdaRoKappaIndividual>() );
    individual.setValue( 0, 2d );
    population.add( individual );
    fitness.reinitialize(population);
    fit = fitness.getFitness(population.get( 2 ));

    assertTrue( fit <= 0.25  + eps && fit >= 0.25 - eps);
    try {
      Field sum  = fitness_reflection.getDeclaredField("sum");
      sum.setAccessible(true);
      fitness_sum = (double) sum.getDouble( fitness ); 
      assertTrue("Sum has wrong value.",fitness_sum <= 3.5/3d + eps && fitness_sum >= 3.5/3d  - eps);
    } catch( Exception ex) {
      fail(ex.getMessage());
    }
    sum_of_fitnesses = 0;
    for( int i = 0; i < population.size(); i++) {
      fitness.reinitialize(population);
      sum_of_fitnesses += fitness.getFitness( population.get( i ));
    }
    assertTrue("Sum of fitnesses isn't 1.", sum_of_fitnesses >= 1 - eps && sum_of_fitnesses <= 1 + eps);

    /* Testing if the best individual has almost best possible fitness*/
    individual = new EvMiLambdaRoKappaIndividual(1);
    individual.setObjectiveFunction( new EvRealOneMax<EvMiLambdaRoKappaIndividual>() );
    individual.setValue( 0, 1d );
    population.add( individual );
    fitness.reinitialize(population);
    fit = fitness.getFitness( population.get(3));
    assertTrue(fit >= 1d - eps);
    
    sum_of_fitnesses = 0;
    for( int i = 0; i < population.size(); i++) {
      fitness.reinitialize(population);
      sum_of_fitnesses += fitness.getFitness( population.get( i ));
    }
    assertTrue("Sum of fitnesses isn't 1.", sum_of_fitnesses >= 1 - eps && sum_of_fitnesses <= 1 + eps);
  }
}
