
package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorMIMICOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import junit.framework.TestCase;

/**
 * @author Marek Chrusciel, Michal Humenczuk
 * 
 */

public class EvMimicOperatorTest extends TestCase {

  EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>();

  EvObjectiveFunction<EvBinaryVectorIndividual> obj_function = new EvOneMax();

  protected void setUp() throws Exception {
    super.setUp();
    EvBinaryVectorIndividual b = new EvBinaryVectorIndividual( 2 );
    b.setGene( 0, 1 );
    b.setGene( 1, 0 );
    population.add( b );

    b = new EvBinaryVectorIndividual( 2 );
    b.setGene( 0, 0 );
    b.setGene( 1, 0 );
    population.add( b );

    b = new EvBinaryVectorIndividual( 2 );
    b.setGene( 0, 1 );
    b.setGene( 1, 0 );
    population.add( b );

    population.setObjectiveFunction( obj_function );
  }

  /** test if method getColumn form population is working crrectly */
  @SuppressWarnings("unchecked")
  public void testGetColumn() {
    Class mimic = EvBinaryVectorMIMICOperator.class;
    Class[] param = new Class[2];
    param[0] = EvPopulation.class;
    param[1] = int.class;
    Method getColumn;

    try {
      getColumn = mimic.getDeclaredMethod( "getColumn", param );
      getColumn.setAccessible( true );
      EvBinaryVectorMIMICOperator mop = new EvBinaryVectorMIMICOperator( 10 );
      Object[] args = new Object[2];
      args[0] = population;
      args[1] = 0;

      Vector<Boolean> result = (Vector<Boolean>) getColumn.invoke( mop, args );
      Vector<Boolean> expected = new Vector<Boolean>();
      expected.add( true );
      expected.add( false );
      expected.add( true );

      assertEquals( expected, result );
    } catch ( AssertionError e ) {
      fail( e.getMessage() );

    } catch ( Exception e ) {
      fail( e.getMessage() );
    }
  }

  /** test if entropy is evaluated correctly */
  @SuppressWarnings("unchecked")
  public void testGetEntrophy() {
    Class<EvBinaryVectorMIMICOperator> mimic = EvBinaryVectorMIMICOperator.class;
    Class[] param = new Class[1];
    param[0] = Vector.class;

    Method getEntrophy;

    try {
      getEntrophy = mimic.getDeclaredMethod( "getEntrophy", param );
      getEntrophy.setAccessible( true );

      EvBinaryVectorMIMICOperator mop = new EvBinaryVectorMIMICOperator( 10 );
      Object[] args = new Object[1];
      Vector<Boolean> column = new Vector<Boolean>();
      column.add( true );
      column.add( false );
      column.add( true );
      column.add( true );
      column.add( true );
      column.add( false );
      args[0] = column;

      double result = (Double) getEntrophy.invoke( mop, args );

      if ( result <= 0 )
        throw new Exception( "getEntropy method invalid" );

    } catch ( Exception e ) {
      fail( e.getMessage() );
    }
  }

  public void testGetConditionalEntrophy() {

    Class<EvBinaryVectorMIMICOperator> mimic = EvBinaryVectorMIMICOperator.class;
    Class[] param = new Class[2];
    param[0] = Vector.class;
    param[1] = Vector.class;

    Method getConditionEntrophy;

    try {
      getConditionEntrophy = mimic.getDeclaredMethod( "getConditionalEntrophy", param );
      getConditionEntrophy.setAccessible( true );

      EvBinaryVectorMIMICOperator mop = new EvBinaryVectorMIMICOperator( 10 );
      Object[] args = new Object[2];
      Vector<Boolean> X = new Vector<Boolean>();
      Vector<Boolean> Y = new Vector<Boolean>();
      X.add( true );
      X.add( false );
      X.add( true );

      Y.add( true );
      Y.add( true );
      Y.add( false );
      args[0] = X;
      args[1] = Y;

      double result = (Double) getConditionEntrophy.invoke( mop, args );

      if ( result <= 0 )
        throw new Exception( "getConditionalEntropy method invalid" );
    } catch ( Exception e ) {
      e.printStackTrace();
      fail( e.getMessage() );

    }
  }

  public void testGetColumnWithMaxEntrophy() {
    Class<EvBinaryVectorMIMICOperator> mimic = EvBinaryVectorMIMICOperator.class;
    Class[] param = new Class[1];
    param[0] = EvPopulation.class;

    Method getColumnWithMaxEntrophy;

    try {
      getColumnWithMaxEntrophy = mimic.getDeclaredMethod( "getColumnWithMaxEntrophy", param );
      getColumnWithMaxEntrophy.setAccessible( true );

      EvBinaryVectorMIMICOperator mop = new EvBinaryVectorMIMICOperator( 10 );

      Object[] args = new Object[1];
      args[0] = population;

      int result = (Integer) getColumnWithMaxEntrophy.invoke( mop, args );

      assertTrue( result >= 0 && result < population.get( 0 ).getDimension() );

    } catch ( AssertionError e ) {
      fail( e.getMessage() );
    } catch ( Exception e ) {
      fail( e.getMessage() );
    }
  }

  public void testGetColumnWithMaxConditionalEntrophy() {
    Class<EvBinaryVectorMIMICOperator> mimic = EvBinaryVectorMIMICOperator.class;
    Class[] param = new Class[3];
    param[0] = EvPopulation.class;
    param[1] = Integer.class;
    param[2] = ArrayList.class;

    Method getColumnWithMaxConditionalEntrophy;

    try {
      getColumnWithMaxConditionalEntrophy = mimic.getDeclaredMethod(
          "getColumnWithMaxConditionalEntrophy", param );
      getColumnWithMaxConditionalEntrophy.setAccessible( true );

      EvBinaryVectorMIMICOperator mop = new EvBinaryVectorMIMICOperator( 10 );

      ArrayList<Integer> array_param = new ArrayList<Integer>(1);
      array_param.add( 0 );
      
      Object[] args = new Object[3];
      args[0] = population;
      args[1] = 1;
      args[2] = array_param;
    
      int result = (Integer) getColumnWithMaxConditionalEntrophy.invoke( mop, args );

      assertTrue( result >= 0 && result < population.get( 0 ).getDimension() );

    } catch ( AssertionError e ) {
      e.printStackTrace();
      fail( e.getMessage() );
    } catch ( Exception e ) {
      e.printStackTrace();
      fail( e.getMessage() );
    }
  }

  public void testEvaluateEntrophiesPermutation() {
    Class<EvBinaryVectorMIMICOperator> mimic = EvBinaryVectorMIMICOperator.class;
    Class[] param = new Class[1];
    param[0] = EvPopulation.class;

    Method evaluateEntrophiesPermutation;
    Field permutation;

    try {
      evaluateEntrophiesPermutation = mimic.getDeclaredMethod( "evaluateEntrophiesPermutation",
          param );
      evaluateEntrophiesPermutation.setAccessible( true );
      Object[] args = new Object[1];
      args[0] = population;

      EvBinaryVectorMIMICOperator mop = new EvBinaryVectorMIMICOperator( 10 );
      evaluateEntrophiesPermutation.invoke( mop, args );

      permutation = mimic.getDeclaredField( "permutation" );
      permutation.setAccessible( true );

      int[] result = (int[]) permutation.get( mop );

      Arrays.sort( result );

      for( int i = 0; i < result.length; i++ ) {
        assertEquals( i, result[i] );
      }

    } catch ( AssertionError e ) {
      e.printStackTrace();
      fail( e.getMessage() );
    } catch ( Exception e ) {
      e.printStackTrace();
      fail( e.getMessage() );
    }

  }

  public void testInitPropabilitiesVector() {
    Class<EvBinaryVectorMIMICOperator> mimic = EvBinaryVectorMIMICOperator.class;
    Class[] param = new Class[1];
    param[0] = EvPopulation.class;

    Method initPropabilitiesVector;
    Method evaluateEntrophiesPermutation;
    Field first_permutation_element_one_probability;
    Field one_probabilities_vector;
    Field zero_probabilities_vector;

    EvBinaryVectorMIMICOperator mop = new EvBinaryVectorMIMICOperator( 10 );

    try {
      initPropabilitiesVector = mimic.getDeclaredMethod( "initPropabilitiesVector", param );
      evaluateEntrophiesPermutation = mimic.getDeclaredMethod( "evaluateEntrophiesPermutation",
          param );

      first_permutation_element_one_probability = mimic
          .getDeclaredField( "first_permutation_element_one_probability" );
      one_probabilities_vector = mimic.getDeclaredField( "one_probabilities_vector" );
      zero_probabilities_vector = mimic.getDeclaredField( "zero_probabilities_vector" );

      initPropabilitiesVector.setAccessible( true );
      evaluateEntrophiesPermutation.setAccessible( true );
      first_permutation_element_one_probability.setAccessible( true );
      one_probabilities_vector.setAccessible( true );
      zero_probabilities_vector.setAccessible( true );

      Object[] args = new Object[1];
      args[0] = population;

      evaluateEntrophiesPermutation.invoke( mop, args );
      initPropabilitiesVector.invoke( mop, args );

      Double first_prob = (Double) first_permutation_element_one_probability.get( mop );
      double[] probs_one = (double[]) one_probabilities_vector.get( mop );
      double[] probs_zero = (double[]) zero_probabilities_vector.get( mop );

      int dimension = population.get( 0 ).getDimension();

      assertTrue( first_prob >= 0 && first_prob <= 1 );

      assertEquals( dimension, probs_one.length );
      assertEquals( dimension, probs_zero.length );

      for( Double prob : probs_zero ) {
        assertTrue( prob >= 0 && prob <= 1 );
      }
      for( Double prob : probs_one ) {
        assertTrue( prob >= 0 && prob <= 1 );
      }

    } catch ( AssertionError e ) {
      e.printStackTrace();
      fail( e.getMessage() );
    } catch ( Exception e ) {
      e.printStackTrace();
      fail( e.getMessage() );
    }
  }

  public void testGenerateIndividual() {
    Class<EvBinaryVectorMIMICOperator> mimic = EvBinaryVectorMIMICOperator.class;
    Class[] param = new Class[1];
    param[0] = EvPopulation.class;

    Method initPropabilitiesVector;
    Method evaluateEntrophiesPermutation;

    Class[] obj_fun_param = new Class[1];
    obj_fun_param[0] = EvObjectiveFunction.class;

    Method generateIndividual;

    EvBinaryVectorMIMICOperator mop = new EvBinaryVectorMIMICOperator( 10 );

    try {
      initPropabilitiesVector = mimic.getDeclaredMethod( "initPropabilitiesVector", param );
      evaluateEntrophiesPermutation = mimic.getDeclaredMethod( "evaluateEntrophiesPermutation",
          param );
      generateIndividual = mimic.getDeclaredMethod( "generateIndividual", obj_fun_param );

      generateIndividual.setAccessible( true );
      initPropabilitiesVector.setAccessible( true );
      evaluateEntrophiesPermutation.setAccessible( true );

      Object[] args = new Object[1];
      args[0] = population;

      Object[] obj_args = new Object[1];
      obj_args[0] = obj_function;

      evaluateEntrophiesPermutation.invoke( mop, args );
      initPropabilitiesVector.invoke( mop, args );

      EvBinaryVectorIndividual individual = (EvBinaryVectorIndividual) generateIndividual.invoke( mop, obj_args );
      EvObjectiveFunction result_obj_function = individual.getObjectiveFunction();

      assertEquals( obj_function.getClass(), result_obj_function.getClass() );

    } catch ( AssertionError e ) {
      e.printStackTrace();
      fail( e.getMessage() );
    } catch ( Exception e ) {
      e.printStackTrace();
      fail( e.getMessage() );
    }
  }

}
