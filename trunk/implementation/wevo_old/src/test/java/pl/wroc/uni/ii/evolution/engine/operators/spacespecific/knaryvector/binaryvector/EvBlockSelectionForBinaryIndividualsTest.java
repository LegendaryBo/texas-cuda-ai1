package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;
/**
 * @author: Piotr Baraniak, Marek Chru�ciel 
 */

import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvBlockSelection;

import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

public class EvBlockSelectionForBinaryIndividualsTest extends TestCase {
  EvPopulation<EvBinaryVectorIndividual> population;
  Random random;
  protected void setUp() {
    int magic_number = 10;
    random = new Random();
    population = new EvPopulation<EvBinaryVectorIndividual>();
    
    for(int i = 0; i < magic_number; i++) {
      EvBinaryVectorIndividual indiv = new EvBinaryVectorIndividual(magic_number);
      for(int j = 0; j < magic_number; j++) {
        indiv.setGene(i, random.nextInt(2));
      }
      indiv.setObjectiveFunction(new EvOneMax()); // added by to make the test pass, Kamil Dworakowski
      population.add(indiv);
    }
  }
  
  /* Test for M=2, 10 individuals, and we check if the worst two individuals are in result*/
  public void testIfAllWasCorrect(){
   
    EvBlockSelection<EvBinaryVectorIndividual> selector = 
      new EvBlockSelection<EvBinaryVectorIndividual>(2);
    EvPopulation<EvBinaryVectorIndividual> population_after_selection = selector.apply(population);
    population.sort();
    for(int i = 0; i < population_after_selection.size(); i++) {
      for(int j = 0; j < 2; j++)
        assertNotSame("Worst individual stay.", population_after_selection.get(i), population.get(j));
    }
  }
}
