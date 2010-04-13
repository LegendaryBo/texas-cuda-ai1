package pl.wroc.uni.ii.evolution.engine.individuals;
/**
 * @author Piotr Baraniak
 */

import java.util.Collections;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvBinaryPattern;

public class EvPopulationTest extends TestCase {

  @SuppressWarnings("unchecked")
  public void testIfHashWorkWell() {
    EvPopulation<EvIndividual> population_first = new EvPopulation<EvIndividual>();
    EvPopulation<EvIndividual> population_second = new EvPopulation<EvIndividual>();
    assertNotSame("there are same hash in two random populations",
        population_first.getHash(), population_second.getHash());
    population_second = population_first.clone();
    assertEquals("Hashes aren't same in original and clone",
        population_first.getHash(),population_second.getHash());
    long temp = population_first.getHash();
    assertTrue(population_first.add(new EvBinaryVectorIndividual(1)));
    assertNotSame("Population have changed and hash no", temp,population_first.getHash());
    EvPopulation<EvIndividual> test_individuals = new EvPopulation<EvIndividual>();
    
    /*Creating simple individuals*/
    int[] pattern = {1, 0};
    int[] tmp = {1, 0};
    EvBinaryVectorIndividual temp_individual = new EvBinaryVectorIndividual(tmp);
    temp_individual.setObjectiveFunction(new EvBinaryPattern(pattern));
    test_individuals.add( temp_individual);
    tmp[0] = 0;
    temp_individual = new EvBinaryVectorIndividual(tmp);
    temp_individual.setObjectiveFunction(new EvBinaryPattern(pattern));
    test_individuals.add(temp_individual);
    tmp[1] = 1;
    temp_individual = new EvBinaryVectorIndividual(tmp);
    temp_individual.setObjectiveFunction(new EvBinaryPattern(pattern));
    test_individuals.add(temp_individual);
    tmp[0] = 1;
    temp_individual = new EvBinaryVectorIndividual(tmp);
    temp_individual.setObjectiveFunction(new EvBinaryPattern(pattern));
    test_individuals.add(temp_individual);
    
    population_first = new EvPopulation<EvIndividual>(test_individuals);
    population_second =population_first.clone();
    /* Test if clone work well.*/
    assertTrue(population_first.size() == 4);
    for(int i = 0; i < population_first.size(); i++) {
      assertNotSame("Clone not done well in " + i + " individual.", population_first.get(i), population_second.get(i));
      assertEquals(population_first.get(i).getObjectiveFunctionValue() , population_second.get(i).getObjectiveFunctionValue());
    }
    /* Test methods which potencially could change hash. */
    population_first.sort();
    assertEquals("Hash had changed when sort",population_first.getHash(),population_second.getHash());
    Collections.max(population_first);
    assertEquals("Hash had changed while looking for max",population_first.getHash(),population_second.getHash());
    population_first.reverse();
    assertEquals("Hash had changed while reversing",population_first.getHash(),population_second.getHash());
    

  }
}
