package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.omega;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.omega.EvOmegaMaxFixedPointsNumber;


/**
 * Test for EvOmegaSelection
 * 
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaSelectionTest extends TestCase {

  public void testSelection() {
    EvOmegaIndividual individual = new EvOmegaIndividual(50,null);
    EvPopulation<EvOmegaIndividual> population = 
      new EvPopulation<EvOmegaIndividual>();
      
    for(int i = 0; i < 100; i++) {
      population.add(new EvOmegaIndividual(50, null,25));
    }
      
    for( EvOmegaIndividual  ind : population) {
      ind.setObjectiveFunction(new EvOmegaMaxFixedPointsNumber());
      ind.setTemplate(individual);
    }      
      
    EvOmegaSelection selection = new EvOmegaSelection(10,10);
    EvPopulation<EvOmegaIndividual> result = selection.apply(population);
     
    assertTrue(result.size() == 10); 
  }
}
