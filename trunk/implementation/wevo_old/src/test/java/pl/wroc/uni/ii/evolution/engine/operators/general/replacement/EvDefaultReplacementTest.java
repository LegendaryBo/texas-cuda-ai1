package pl.wroc.uni.ii.evolution.engine.operators.general.replacement;

import java.util.Collections;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvNBestParentPromotedReplacement;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;
import junit.framework.TestCase;

public class EvDefaultReplacementTest extends TestCase {
  
  public void testReturnsTheSameNumberOfIndividualsAsThereAreParentsGiven() throws Exception {
    EvPopulation<EvIndividual> parents = aPopulationWithTwoIndividuals();
    EvPopulation<EvIndividual> children = aPopulationWithThreeIndividuals();
    
    EvNBestParentPromotedReplacement<EvIndividual> replacement = new EvNBestParentPromotedReplacement<EvIndividual>(1);
    
    assertEquals(2,replacement.apply(parents, children).size());
  }
  
  public void testInitializedWithZeroDoesNotPromoteParents() throws Exception {
    EvPopulation<EvIndividual> parents = aPopulationWithTwoIndividuals();
    EvPopulation<EvIndividual> children = aPopulationWithThreeIndividuals();
    
    EvNBestParentPromotedReplacement<EvIndividual> replacement = new EvNBestParentPromotedReplacement<EvIndividual>(0);
    
    assertTrue(Collections.disjoint(parents, replacement.apply(parents, children)));
  }
  
  public void testBestParentsGetsPromotedByOperatorInitializedWith1() throws Exception {
    EvPopulation<EvIndividual> parents = aPopulationWithTwoIndividuals();
    EvPopulation<EvIndividual> children = aPopulationWithThreeIndividuals();
    
    EvBinaryVectorIndividual optimal = aOptimalIndividual();
    parents.add(optimal);
    
    EvPopulation<EvIndividual> new_population = 
      new EvNBestParentPromotedReplacement<EvIndividual>(1).apply(parents, children);
    
    assertTrue(new_population.contains(optimal));
    parents.remove(optimal);
    assertTrue("only 1 parent gets promoted", Collections.disjoint( parents , new_population ));
    assertEquals(3,new_population.size());
  }

  private EvBinaryVectorIndividual aOptimalIndividual() {
    EvBinaryVectorIndividual evBinaryIndividual = anIndividual();
    for(int i = 0; i<evBinaryIndividual.getDimension(); i++)
      evBinaryIndividual.setGene(i, 1);
    evBinaryIndividual.setObjectiveFunction(new EvOneMax());
    return evBinaryIndividual;
  }

  private EvPopulation<EvIndividual> aPopulationWithThreeIndividuals() {
    EvPopulation<EvIndividual> population = aPopulationWithTwoIndividuals();
    population.add(anIndividual());
    return population;
  }

  private EvBinaryVectorIndividual anIndividual() {
    EvBinaryVectorIndividual evBinaryIndividual = new EvBinaryVectorIndividual(EvRandomizer.INSTANCE.nextBooleanListAsInt(32, 15));
    evBinaryIndividual.setObjectiveFunction(new EvOneMax());
    return evBinaryIndividual;
  }

  private EvPopulation<EvIndividual> aPopulationWithTwoIndividuals() {
    EvPopulation<EvIndividual> parents = new EvPopulation<EvIndividual>();
    parents.add(anIndividual());
    parents.add(anIndividual());
    return parents;
  }

}
