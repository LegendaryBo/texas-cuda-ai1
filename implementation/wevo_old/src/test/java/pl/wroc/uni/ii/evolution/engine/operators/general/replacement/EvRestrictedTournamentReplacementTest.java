package pl.wroc.uni.ii.evolution.engine.operators.general.replacement;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.likeness.EvHammingDistanceLikenes;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

/**
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvRestrictedTournamentReplacementTest extends TestCase {

  /**
   * Test pass if child1 is put into parent population and 
   * when child2 isn't.
   */
  public void testApply() {
    
    EvRestrictedTournamentReplacement<EvBinaryVectorIndividual> tournament =
        new EvRestrictedTournamentReplacement<EvBinaryVectorIndividual>(
            4, new EvHammingDistanceLikenes<EvBinaryVectorIndividual>());
    
    EvOneMax one_max = new EvOneMax();
    
    EvPopulation<EvBinaryVectorIndividual> parents = 
        new EvPopulation<EvBinaryVectorIndividual>();
 
    EvPopulation<EvBinaryVectorIndividual> children = 
      new EvPopulation<EvBinaryVectorIndividual>();    
    
    EvBinaryVectorIndividual parent1 = 
        new EvBinaryVectorIndividual(new int[]{1, 1, 1, 0, 0});
    EvBinaryVectorIndividual parent2 = 
      new EvBinaryVectorIndividual(new int[]{1, 1, 1, 1, 0});
    EvBinaryVectorIndividual parent3 = 
      new EvBinaryVectorIndividual(new int[]{0, 1, 1, 0, 0});
    EvBinaryVectorIndividual parent4 = 
      new EvBinaryVectorIndividual(new int[]{1, 0, 1, 0, 0});    

    EvBinaryVectorIndividual child1 = 
      new EvBinaryVectorIndividual(new int[]{1, 1, 1, 1, 1}); 
    EvBinaryVectorIndividual child2 = 
      new EvBinaryVectorIndividual(new int[]{0, 0, 0, 0, 0});     
    
    parent1.setObjectiveFunction(one_max);
    parent2.setObjectiveFunction(one_max);
    parent3.setObjectiveFunction(one_max);
    parent4.setObjectiveFunction(one_max);
    
    child1.setObjectiveFunction(one_max);
    child2.setObjectiveFunction(one_max);
    
    parents.add(parent1);
    parents.add(parent2);
    parents.add(parent3);
    parents.add(parent4);
    
    children.add(child1);
    children.add(child2);
    
    EvPopulation<EvBinaryVectorIndividual> result = 
      tournament.apply(parents, children);   
    
    boolean flag1 = false;
    boolean flag2 = false;
    for (EvBinaryVectorIndividual ind : result) {

      if (ind.equals(child1)) {
        flag1 = true;
      }
      if (ind.equals(child2)) {
        flag2 = true;
      }     
      
    }
    
    assertTrue(flag1);
    assertFalse(flag2);
  }
  
}
