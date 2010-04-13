package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic.EvDiscreteVectorMIMICBayesianNetwork;


/**
 * Tests for MIMIC Operator.
 * 
 * NOT YET COMPLETE !!
 * 
 * @author Grzegorz Lisowski (grzegorz.lisowski@interia.pl)
 * 
 */
public class EvKnaryMIMICTest {

  /**
   * Test population 0.
   */
  private EvPopulation<EvKnaryIndividual> pop0;
  
  
  /**
   * Test network for population 0.
   */
  private EvDiscreteVectorMIMICBayesianNetwork network0;
  
  
  /**
   * Sets up test environment.
   * 
   * @throws java.lang.Exception Exception
   */
  @Before
  public void setUp() throws Exception {
    int[] chromosome0 = {0, 0, 0, 0};
    //int[] chromosome1 = {1, 1, 1, 1};
    int pop0_size = 20;
    //int pop1_size = 15;
    int[] geneValues = {0, 1};
    
    this.network0 = new EvDiscreteVectorMIMICBayesianNetwork(4, geneValues);
    this.pop0 = new EvPopulation<EvKnaryIndividual>(pop0_size);
    
    for (int i = 0; i < pop0_size; i++) {
      pop0.add(new EvKnaryIndividual(chromosome0, Integer.MAX_VALUE));
    }
  }
  
  
  /**
   * Tests network.
   */
  @Test
  public void testNetwork() {
    network0.estimateProbabilities(pop0);
    //System.out.println(network0);
    assertTrue(true);
  }
    
  
  
}
