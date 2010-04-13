/**
 * 
 */
package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryUMDAProbability;

/**
 * Tests for EvbinaryUMDAProbability.
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 *
 */
public class EvBinaryUMDAProbabilityTest {
  //off VisibilityModifier
  //po co mam robic gettery i settery do testow???????????
  /** Population with Individuals with only 0. */
  EvPopulation<EvBinaryVectorIndividual> populationZeros;
  /** Population with Individuals with only 1. */
  EvPopulation<EvBinaryVectorIndividual> populationOnes;
  /** Sample population. */ 
  EvPopulation<EvBinaryVectorIndividual> population;
  /** Expected probability vector for populationZeros. */
  double[] probabilityZeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  /** Expected probability vector for populationOnes. */
  double[] probabilityOnes = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  /** Expected probability vector for population. */
  double[] probability = {0, 0.2, 0.4, 0.6, 0.8, 1, 0.2, 0.4, 0.6, 0.6};
  
  /** Probability operator for testing. */
  EvBinaryUMDAProbability probabilityOperator;
  //on VisibilityModifier
  /**
   * Settings up variables.
   * @throws java.lang.Exception Exception
   */
  @Before
  public void setUp() throws Exception {
    int[] zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int[] ones = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    this.populationZeros = new EvPopulation<EvBinaryVectorIndividual>(39);
    for (int i = 0; i < 39; i++) {
      EvBinaryVectorIndividual in = new EvBinaryVectorIndividual(zeros);
      populationZeros.add(in);
    }
    
    this.populationOnes = new EvPopulation<EvBinaryVectorIndividual>(16);
    for (int i = 0; i < 16; i++) {
      EvBinaryVectorIndividual in = new EvBinaryVectorIndividual(ones);
      populationOnes.add(in);
    }
    
    int[] gen1 = {0, 1, 1, 1, 1, 1, 0, 0, 0, 1};
    int[] gen2 = {0, 0, 1, 1, 1, 1, 0, 0, 1, 0};
    int[] gen3 = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1};
    int[] gen4 = {0, 0, 0, 0, 1, 1, 0, 0, 1, 1};
    int[] gen5 = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0};
    this.population = new EvPopulation<EvBinaryVectorIndividual>(5);
    EvBinaryVectorIndividual in1 = new EvBinaryVectorIndividual(gen1);
    EvBinaryVectorIndividual in2 = new EvBinaryVectorIndividual(gen2);
    EvBinaryVectorIndividual in3 = new EvBinaryVectorIndividual(gen3);
    EvBinaryVectorIndividual in4 = new EvBinaryVectorIndividual(gen4);
    EvBinaryVectorIndividual in5 = new EvBinaryVectorIndividual(gen5);
    this.population.add(in1);
    this.population.add(in2);
    this.population.add(in3);
    this.population.add(in4);
    this.population.add(in5);
    
    //In this case we don't care how big is going to be new population.
    this.probabilityOperator = new EvBinaryUMDAProbability(23);
  }

  /**
   * Testing if probability vector is ok.
   */
  @Test
  public void testComputeProbability() {
    
    check(probabilityOperator.computeProbability(this.populationZeros),
        this.probabilityZeros, "Checking for zeros.");
    check(probabilityOperator.computeProbability(this.populationOnes),
        this.probabilityOnes, "Checking for ones.");
    check(probabilityOperator.computeProbability(this.population),
        this.probability, "Checking for sample population.");
  }

  /**
   * Check if two arrays are equals.
   * @param computed Computed probability vector.
   * @param expected Expected proability vector.
   * @param message Message if test fails.
   */
  private void check(final double[] computed, final double[] expected,
      final String message) {
    if (computed.length !=  expected.length) {
      assertTrue(message + " - computed length=" + computed.length 
          + " expected length=" + expected.length, false);
    }
    boolean isOk = true;
    for (int i = 0; i < computed.length; i++) {
      if (computed[i] != expected[i]) {
        isOk = false;
        System.err.println(computed[i] + " - " + expected[i]);
        //break;
      }
    }
    assertTrue(message, isOk);
  }

}
