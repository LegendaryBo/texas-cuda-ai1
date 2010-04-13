package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector;



import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorOnePointCrossover;

/**
 * Check if the operator mix individuals and if results are
 * casted to correct class.
 * 
 * @author Piotr Baraniak
 * @author Marek Chruœciel 
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvOnePointCrossoverTest extends TestCase {

  List<EvBinaryVectorIndividual> parents;
  int dimension;
  /**
   * Creation of 2 Individuals, length 5, one has true vector, and other false.
   */
  protected void setUp() throws Exception {
    super.setUp();
    dimension = 5;
    parents = new ArrayList<EvBinaryVectorIndividual>(2);
    parents.add( new EvBinaryVectorIndividual(dimension));
    parents.add( new EvBinaryVectorIndividual(dimension));
    for( int i = 0; i < dimension; i++) {
      parents.get(0).setGene(i, 1);
      parents.get(1).setGene(i, 0);
    }
  }
  /**
   * Test if children aren't the same, and there was a breakpoint in crossover.
   * Also test if it works with binary individuals
   */
  public void testOnePointCrossover() {
    EvKnaryVectorOnePointCrossover<EvBinaryVectorIndividual> cross = 
      new EvKnaryVectorOnePointCrossover<EvBinaryVectorIndividual>();
    List<EvBinaryVectorIndividual> children = cross.combine(parents);
    
    
    boolean[] temp = {false, false};
    for(int i = 0; i < dimension; i++) {
      assertNotSame("Children are the same", children.get(0).getGene(i), children.get(1).getGene(i));
      temp[0] = temp[0] || children.get(0).getGene(i) == 1;
      temp[1] = temp[1] || children.get(1).getGene(i) == 1;
    }
    
    assertTrue("There was no break point",temp[0] && temp[1]);
    // check if result are casted to correct class;
    assertNotSame(children.get(0).getClass(), EvKnaryIndividual.class);
    assertEquals(children.get(0).getClass(), EvBinaryVectorIndividual.class);
    
  }
  
  // test if the operator works when applied to KnaryIndividuals
  // and everything is ok. with generics
  public void testknarySolutionSpaces() {
    
    List<EvKnaryIndividual> parents_list = new ArrayList<EvKnaryIndividual>();
    EvKnaryVectorOnePointCrossover<EvKnaryIndividual> cross_knary = 
      new EvKnaryVectorOnePointCrossover<EvKnaryIndividual>();
    
    EvKnaryIndividual ind1 = new EvKnaryIndividual(3, 10);
    EvKnaryIndividual ind2 = new EvKnaryIndividual(3, 10);
    
    // first individual = (2,2,2)
    ind1.setGene(0, 2);
    ind1.setGene(1, 2);
    ind1.setGene(2, 2);

    // second individual = (9,9,9)
    ind2.setGene(0, 9);
    ind2.setGene(1, 9);
    ind2.setGene(2, 9);   
    
    parents_list.add(ind1);
    parents_list.add(ind2);
    
    List<EvKnaryIndividual> children = cross_knary.combine(parents_list);

    // make sure break point exists
    assertNotSame(children.get(0).getGene(0), children.get(1).getGene(0));
     
    // ensure that types are correct correct
    assertEquals(children.get(0).getClass(), EvKnaryIndividual.class);
    assertNotSame(children.get(0).getClass(), EvBinaryVectorIndividual.class);
    
  }

}
