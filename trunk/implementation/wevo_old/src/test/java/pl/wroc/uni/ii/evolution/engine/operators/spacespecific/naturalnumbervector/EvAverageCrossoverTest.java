package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector;


import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.naturalnumbervector.EvNaturalNumberVectorAverageCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvPersistPopulationSizeApplyStrategy;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvSimpleApplyStrategy;

import pl.wroc.uni.ii.evolution.objectivefunctions.naturalnumbervector.EvNaturalPattern;

/**
 * @author Kamil Dworakowski, Jarek Fuks
 */
public class EvAverageCrossoverTest extends TestCase {

  public void testCrossover() throws Exception {
    
    EvNaturalNumberVectorIndividual indiv1 = new EvNaturalNumberVectorIndividual(new int[] { 1, 4 });
    indiv1.setObjectiveFunction(new EvNaturalPattern(new int[] { 2, 2 }));
    EvNaturalNumberVectorIndividual indiv2 = new EvNaturalNumberVectorIndividual(new int[] { 3, 0 });
    indiv2.setObjectiveFunction(new EvNaturalPattern(new int[] { 2, 2 }));
    
    EvNaturalNumberVectorAverageCrossover operator = new EvNaturalNumberVectorAverageCrossover(2);

    EvPopulation<EvNaturalNumberVectorIndividual> parents = 
      new EvPopulation<EvNaturalNumberVectorIndividual>();
    parents.add(indiv1);
    parents.add(indiv2);

    EvNaturalNumberVectorIndividual baby = operator.apply(parents).get(0);

    assertEquals(2, baby.getNumberAtPosition(0));
    assertEquals(2, baby.getNumberAtPosition(1));

    assertEquals(2d, baby.getObjectiveFunctionValue(),0.000000001);
    
   
  }
  
  public void testPopulationOfThree() throws Exception {
    EvPopulation<EvNaturalNumberVectorIndividual> parents =
      new EvPopulation<EvNaturalNumberVectorIndividual>();
    parents.add(new EvNaturalNumberVectorIndividual(new int[] {1,1}));
    parents.add(new EvNaturalNumberVectorIndividual(new int[] {2,4}));
    parents.add(new EvNaturalNumberVectorIndividual(new int[] {3,4}));
    
    parents.setObjectiveFunction(new EvNaturalPattern(new int[] {1,2}));
    
    EvCrossover<EvNaturalNumberVectorIndividual> operator = new EvNaturalNumberVectorAverageCrossover(3);
    operator.setCrossoverStrategy(new EvPersistPopulationSizeApplyStrategy());
    
    EvPopulation<EvNaturalNumberVectorIndividual> pop = operator.apply(parents);
    assertEquals(3, pop.size());
    
    for (int i = 0; i < 3; i++) {
      
      assertEquals(2, pop.get(i).getNumberAtPosition(0));
      assertEquals(3, pop.get(i).getNumberAtPosition(1));
    }
    
  }
  
  public void testPopulationOfOne() throws Exception {
    EvPopulation<EvNaturalNumberVectorIndividual> parents =
      new EvPopulation<EvNaturalNumberVectorIndividual>();
    parents.add(new EvNaturalNumberVectorIndividual(new int[] {1,1}));
    parents.add(new EvNaturalNumberVectorIndividual(new int[] {2,1}));
    parents.add(new EvNaturalNumberVectorIndividual(new int[] {3,4}));
    
    parents.setObjectiveFunction(new EvNaturalPattern(new int[] {1,2}));
    
    EvCrossover<EvNaturalNumberVectorIndividual> operator = new EvNaturalNumberVectorAverageCrossover(3);
    operator.setCrossoverStrategy(new EvSimpleApplyStrategy());
    
    EvPopulation<EvNaturalNumberVectorIndividual> pop = operator.apply(parents);
    assertEquals(1, pop.size());
    
  }
}
