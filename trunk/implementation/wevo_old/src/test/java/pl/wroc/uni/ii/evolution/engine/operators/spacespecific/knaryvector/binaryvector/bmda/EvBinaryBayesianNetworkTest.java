package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryIndividualTestWithFunction;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.bayesnetwork.EvBinaryBayesianNetwork;

/**
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvBinaryBayesianNetworkTest extends TestCase {

  /**
   * Test cloning
   */
  public void testClone() {
    
    EvBinaryBayesianNetwork network = new EvBinaryBayesianNetwork(3);
    EvBinaryBayesianNetwork clone = null;
    
    // blank population
    EvPopulation<EvBinaryVectorIndividual> population = 
        new EvPopulation<EvBinaryVectorIndividual>();
    
    population.add(new EvBinaryVectorIndividual(4));
    
    network.initialize(population);
    
    network.addEdge(0, 3);
    network.addEdge(1, 3);
    network.addEdge(2, 3);
    
    clone = network.clone();
    
    for (int i = 0; i < network.getSize(); i++) {
      int[] parents = network.getParentsIndexes(i);
      int[] parents_clone = clone.getParentsIndexes(i);
      System.out.println(parents.length + " " + parents_clone.length);
      
      if (parents.length != parents_clone.length) {
        fail();
      }
      
      for (int j = 0; j < parents.length; j++) {
        if (parents[j] != parents_clone[j]) {
            fail();
        }
      }
    }

    for (int i = 0; i < network.getSize(); i++) {
      int[] parents = network.getParentsIndexes(i);
      int[] parents_clone = clone.getParentsIndexes(i);
      
      for (int j = 0; j < parents_clone.length; j++) {
        if (parents[j] != parents_clone[j]) {
            fail();
        }
      }
    }    
    
    
  }
  
  
  /**
   * Test if network generates correct individuals.
   * Also test if probabilities are correct.
   */
  public void testGenerate() {
  
    EvBinaryBayesianNetwork network = new EvBinaryBayesianNetwork(2);
    EvBinaryBayesianNetwork clone = null;
    
    // blank population
    EvPopulation<EvBinaryVectorIndividual> population = 
        new EvPopulation<EvBinaryVectorIndividual>();
    
    population.add(new EvBinaryVectorIndividual(4));
    population.add(new EvBinaryVectorIndividual(
        new int[]{1, 1, 1, 0}));
    population.add(new EvBinaryVectorIndividual(
        new int[]{1, 0, 1, 0}));
    population.add(new EvBinaryVectorIndividual(
        new int[]{0, 1, 1, 0}));  
    
    network.initialize(population);
    
    network.addEdge(0, 2);
    network.addEdge(1, 2);   
    
    EvBinaryVectorIndividual individual = network.generate();
    
    assertEquals(individual.getGene(3), 0);
    
    if (individual.getGene(0) == 0 && individual.getGene(1) == 0) {
      assertTrue(individual.getGene(2) == 0);
    }

    if (individual.getGene(0) == 1 || individual.getGene(1) == 1) {
      assertTrue(individual.getGene(2) == 1);
    }    
    
    System.out.println(individual);
    
    
    
  }
  
}
