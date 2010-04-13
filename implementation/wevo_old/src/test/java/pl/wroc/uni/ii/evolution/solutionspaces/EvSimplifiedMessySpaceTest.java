package pl.wroc.uni.ii.evolution.solutionspaces;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyMaxSum;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyObjectiveFunction;
import pl.wroc.uni.ii.evolution.solutionspaces.EvSimplifiedMessySpace;

public class EvSimplifiedMessySpaceTest extends TestCase {

  public void testBelongsTo() {
    
    EvSimplifiedMessyObjectiveFunction dummy = new EvSimplifiedMessyObjectiveFunction(5, new EvSimplifiedMessyMaxSum(), 10);
    
    EvSimplifiedMessyIndividual ind1 = new EvSimplifiedMessyIndividual(10);
   
    EvSimplifiedMessySpace space1 = new EvSimplifiedMessySpace(dummy, 10);
    EvSimplifiedMessySpace space2 = new EvSimplifiedMessySpace(dummy, 9);
    
    assertTrue(space1.belongsTo(ind1));
    assertFalse(space2.belongsTo(ind1));
    
    
    
    EvSimplifiedMessyIndividual ind2 = new EvSimplifiedMessyIndividual(10);
    
    ind2.setGeneValue(0, 2);
    ind2.setGeneValue(1, 1);
    ind2.setGeneValue(2, 5);
    ind2.setGeneValue(2, 6);
    
    EvSimplifiedMessySpace space3 = new EvSimplifiedMessySpace(dummy, 10, 2);
    EvSimplifiedMessySpace space4 = new EvSimplifiedMessySpace(dummy, 10, 6);
    EvSimplifiedMessySpace space5 =  new EvSimplifiedMessySpace(dummy, 10, 5);
    assertTrue(space4.belongsTo(ind2));
    assertFalse(space3.belongsTo(ind2));
    assertFalse(space5.belongsTo(ind2));
    
  }  

}
