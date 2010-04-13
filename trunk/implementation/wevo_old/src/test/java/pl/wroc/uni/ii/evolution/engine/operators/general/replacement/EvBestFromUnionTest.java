package pl.wroc.uni.ii.evolution.engine.operators.general.replacement;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;
import junit.framework.TestCase;

public class EvBestFromUnionTest extends TestCase {
  
  public void testReturnsTheNumberOfIndividualsSet() throws Exception {
    EvPopulation<EvIndividual> parents = aPopulationWithTwoIndividuals();
    EvPopulation<EvIndividual> children = aPopulationWithThreeIndividuals();
    
    EvBestFromUnionReplacement<EvIndividual> replacement = new EvBestFromUnionReplacement<EvIndividual>(3);
    
    assertEquals(3,replacement.apply(parents, children).size());
  }
  
  public void testReturnsTheSameNumberOfIndividualsAsThereAreParentsGiven() throws Exception {
    EvPopulation<EvIndividual> parents = aPopulationWithTwoIndividuals();
    EvPopulation<EvIndividual> children = aPopulationWithThreeIndividuals();
    
    EvBestFromUnionReplacement<EvIndividual> replacement = new EvBestFromUnionReplacement<EvIndividual>();
    
    assertEquals(2,replacement.apply(parents, children).size());
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
