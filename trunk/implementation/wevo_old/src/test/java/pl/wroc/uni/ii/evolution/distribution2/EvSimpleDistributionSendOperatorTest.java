package pl.wroc.uni.ii.evolution.distribution2;

import java.net.MalformedURLException;
import java.net.URL;

import pl.wroc.uni.ii.evolution.distribution2.operators.EvSimpleDistributionGetOperator;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import junit.framework.TestCase;

/**
 * Simple test for EvSimpleDistributionSendOperator. 
 * 
 * @author Karol Asgaroth Stosiek
 *
 */
public class EvSimpleDistributionSendOperatorTest extends TestCase {

  /**
   * Simple constructor test.
   */
  public void testConstructor() {
    
    /* substituting the URL object with our implementation */
    URL testedURL = null;
    
    try {
      testedURL = new URL("http://localhost/");
    
    } catch (MalformedURLException e) {
      e.printStackTrace();
    }
    
    int invalidAmount = -10;
    int validAmount = 10;
    
    @SuppressWarnings("unused")
    EvSimpleDistributionGetOperator<EvKnaryIndividual> operator;
    
    try {
      /* trying to create new operator instance with invalid arguments */
      operator = new EvSimpleDistributionGetOperator<EvKnaryIndividual>(
            testedURL, invalidAmount);
      
    } catch (Exception e) {
      /* this exception is expected to be thrown, since the initial 
       * amount for constructor is invalid */
      assertTrue(true);
    }
    
    try {
      /* trying to create new operator instance with valid arguments */
      operator = new EvSimpleDistributionGetOperator<EvKnaryIndividual>(
            testedURL, validAmount);
      
    } catch (Exception e) {
      /* this exception should not be thrown, since operations
       * above are expected to succeed */
      assertTrue(false);
    }
  }

}
