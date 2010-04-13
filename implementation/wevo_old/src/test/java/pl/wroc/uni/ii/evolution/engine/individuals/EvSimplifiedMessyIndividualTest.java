package pl.wroc.uni.ii.evolution.engine.individuals;

import java.util.ArrayList;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

public class EvSimplifiedMessyIndividualTest extends TestCase {
  public void testClone() {
    EvSimplifiedMessyIndividual individual = new EvSimplifiedMessyIndividual(10);
    individual.addGeneValue(0, 10);
    individual.addGeneValue(0, 13);
    assertTrue(individual.equals(individual.clone()));  
  }
  
  public void testRemoveDuplicateGeneValues() {
   EvSimplifiedMessyIndividual ind1 = new EvSimplifiedMessyIndividual(2);
   
   ArrayList<Integer> gene_values = new ArrayList<Integer>();
   
   gene_values.add(1);
   gene_values.add(1);
   
   ind1.setGeneValues(0, gene_values);
   ind1.setGeneValue(1, 2);
   
   EvSimplifiedMessyIndividual ind2 = ind1.clone();
   ind2.removeAllDuplicateGeneValues();
   EvSimplifiedMessyIndividual ind3 = new EvSimplifiedMessyIndividual(2);
   ind3.setGeneValue(0, 1);
   ind3.setGeneValue(1, 2);
   assertTrue(ind3.equals(ind2));
   
  }


  public void testSetGetGene() {

    EvSimplifiedMessyIndividual individual = new EvSimplifiedMessyIndividual(10);
    int set_value = EvRandomizer.INSTANCE.nextInt(0, individual.getLength());
    
    individual.addGeneValue(3, set_value);
    individual.addGeneValue(3, set_value+1); 
  
    assertEquals(set_value, individual.getGeneValue(3));
    assertEquals(set_value, individual.getGeneValue(3, 0));
    assertEquals(set_value+1, individual.getGeneValue(3,1));
    
  }
  
  
  public void testRemoveGene() {
    EvSimplifiedMessyIndividual individual = new EvSimplifiedMessyIndividual(1);
    individual.setGeneValue(0, 23);
    assertEquals(1, individual.getGeneValues(0).size());
    individual.removeGene(0);
    assertTrue(individual.getGeneValues(0).size() == 0);
  }
  
}
