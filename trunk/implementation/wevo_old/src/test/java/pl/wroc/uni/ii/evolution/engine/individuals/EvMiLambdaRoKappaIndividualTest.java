package pl.wroc.uni.ii.evolution.engine.individuals;

import java.lang.reflect.Field;

import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import junit.framework.TestCase;
/**
 * @author Lukasz Witko, Piotr Baraniak
 */

public class EvMiLambdaRoKappaIndividualTest extends TestCase {

  public void testIndividual(){
    EvMiLambdaRoKappaIndividual individual;
    individual = new EvMiLambdaRoKappaIndividual(5);
    
    Field alpha_field;
    try {
      alpha_field = individual.getClass().getDeclaredField( "alpha" );
      alpha_field.setAccessible( true );
      double[] alpha = (double[]) alpha_field.get( individual );
      assertTrue( alpha.length == 15 );
    } catch ( Exception e ) {
      fail(e.getMessage());
    }
    individual.setAlpha( 0, .1 );
    assertTrue( individual.getAlpha( 0 ) == .1 );

 }
}
