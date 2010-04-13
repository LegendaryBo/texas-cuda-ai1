package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import java.lang.reflect.Field;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.realvector.EvRealOneMax;
import Jama.Matrix;

/**
 * 
 * @author Lukasz Witko, Piotr Baraniak, Tomasz Kozakiewicz
 *
 */
public class EvRotationMutationTest extends TestCase {

  public void testMutation() {
    EvPopulation<EvMiLambdaRoKappaIndividual> population = new EvPopulation<EvMiLambdaRoKappaIndividual>();
    
    EvMiLambdaRoKappaIndividual individual = new EvMiLambdaRoKappaIndividual( new double[] { 3d, 7d, 1d },
        new double[] { 0.69, 0.1, .02 }, new double[] { 3d, 0.2, 0.77, .4, .5, .43 } );

    individual.setObjectiveFunction( new EvRealOneMax<EvMiLambdaRoKappaIndividual>() );
    
    population.add( individual );
    
    double tau_prim;
    double tau;
    
/*    double mi = 10;
    double lambda = mi / 2;
    double k = mi * Math.log( lambda / mi );
    System.out.println("k: " + k);
    double pm = 1;
    double delta = 1 / Math.sqrt( 2 );
    double n = population.get( 0 ).getDimension();
    double nsigma = n;
    
    tau_prim = k / Math.sqrt(pm) * delta / Math.sqrt( n );
    tau = (k / Math.sqrt(pm)) * Math.sqrt( 1 - delta*delta ) / Math.sqrt( n / Math.sqrt( nsigma ) );
*/    
    tau_prim = 1d/8d;
    tau = 1d/6d;
    
    EvMiLambdaRoKappaRotationMutation rotation_mutation = new EvMiLambdaRoKappaRotationMutation( 0.0873, tau, tau_prim  );
    
    EvPopulation<EvMiLambdaRoKappaIndividual> resulted_population;
    
    resulted_population = rotation_mutation.apply( population );
    
    assertNotNull( resulted_population );
  }
  
  public void testLarge() {
    EvPopulation<EvMiLambdaRoKappaIndividual> population = new EvPopulation<EvMiLambdaRoKappaIndividual>();

    int size = 80;
    int alpha_size = size*(size+1)/2;
    double[] val = new double[size];
    double[] sigma = new double[size];
    double[] alpha = new double[alpha_size];
    
    for( int i = 0; i < size; i++ ) {
      val[i] = i;
      sigma[i] = i/(double)size;
    }
    
    for( int i = 0; i < alpha_size; i++ ) {
      alpha[i] = i/(double)size;
    }
    
    EvMiLambdaRoKappaIndividual individual = new EvMiLambdaRoKappaIndividual( val, sigma, alpha);

    individual.setObjectiveFunction( new EvRealOneMax<EvMiLambdaRoKappaIndividual>() );
    
    population.add( individual );
    
  
    double tau_prim;
    double tau;
    tau_prim = 1d/8d;
    tau = 1d/6d;
    
    EvMiLambdaRoKappaRotationMutation rotation_mutation = new EvMiLambdaRoKappaRotationMutation( 0.0873, tau, tau_prim  );
    
    EvPopulation<EvMiLambdaRoKappaIndividual> resulted_population;
    
    resulted_population = rotation_mutation.apply( population );
    assertNotNull( resulted_population );
    try {
      Field matrix_z_field = rotation_mutation.getClass().getDeclaredField( "matrix_z" );
      matrix_z_field.setAccessible( true );
      Matrix matrix_z = (Matrix)matrix_z_field.get( rotation_mutation );

      Field matrix_eps_field = rotation_mutation.getClass().getDeclaredField( "matrix_epsilon" );
      matrix_eps_field.setAccessible( true );
      Matrix matrix_eps = (Matrix)matrix_eps_field.get( rotation_mutation );
      
      assertEquals( matrix_z.normF(), matrix_eps.normF(), .00000001 );
    } catch ( Exception e ) {
      fail(e.getMessage() );
    }

    
  }
}