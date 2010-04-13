package pl.wroc.uni.ii.evolution.engine.individuals;

import junit.framework.TestCase;
import java.util.ArrayList;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;

/**
 * Test of EvOmegaIndividual
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaIndividualTest extends TestCase {
  public void testConstructor() {
    EvOmegaIndividual ind = new EvOmegaIndividual(10, null);
    assertTrue(ind.getChromosomeLength() == 10);
    assertTrue(ind.getExpressedGenesNumber() == 10);
    ind = new EvOmegaIndividual(10,null,5);
    assertTrue(ind.getChromosomeLength() == 5);
    assertTrue(ind.getExpressedGenesNumber() == 5);
    EvOmegaIndividual ind2 = new EvOmegaIndividual(ind);
    assertEquals(ind, ind2);
  }
  
  public void testClone() {
    EvOmegaIndividual ind = new EvOmegaIndividual(10,null);
    assertEquals(ind, ind.clone());
  }
  
  public void testToTemplate() {
    EvOmegaIndividual ind = new EvOmegaIndividual(50, null);
    EvOmegaIndividual ind2 = new EvOmegaIndividual(50, null, 25);
    ind2.setTemplate(ind);
    ind2 = ind2.toTemplate();
    assertEquals(50, ind2.getExpressedGenesNumber());
  }
  
  public void testFenotype() {
    EvOmegaIndividual ind = new EvOmegaIndividual(50, null);
    EvOmegaIndividual ind2 = new EvOmegaIndividual(50, null, 25);
    
    ind2.setTemplate(ind);
    
    ArrayList<Integer> fenotype = ind2.getFenotype();
    ArrayList<Integer> temp = new ArrayList<Integer>(fenotype.size());
    
    for(int i = 0; i < fenotype.size(); i++) {
      temp.add(0);
    }
    
    assertTrue(fenotype.size() == ind.genotype_length);
    
    for(int i = 0; i < fenotype.size(); i++) {
      temp.set(fenotype.get(i), temp.get(fenotype.get(i))+1);
    }
    
    for(int i = 0; i < 50; i++) {
      assertTrue(temp.get(i) == 1);
    }
  }
}
