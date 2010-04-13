package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvSimplifiedMessyGAOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyMaxSum;
import pl.wroc.uni.ii.evolution.objectivefunctions.messy.EvSimplifiedMessyObjectiveFunction;

/**
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvSimplifiedMessyGAOperatorTest extends TestCase {

    public void testOnlyApply() {
  
    // creating individuals  
    EvSimplifiedMessyIndividual individuals[] = { 
        new EvSimplifiedMessyIndividual(2), 
        new EvSimplifiedMessyIndividual(2) 
    };
    individuals[0].addGeneValue(0,2);    
    individuals[0].addGeneValue(1,2);    
    individuals[1].addGeneValue(0,0);    
    individuals[1].addGeneValue(1,0);    
    
    EvPopulation<EvSimplifiedMessyIndividual> population =
        new EvPopulation<EvSimplifiedMessyIndividual>(individuals);

    EvSimplifiedMessyObjectiveFunction messyFn = new EvSimplifiedMessyObjectiveFunction(
        1, new EvSimplifiedMessyMaxSum(), 1);

    population.setObjectiveFunction(messyFn);

    /* selection test
     * in this case selection_population always should have
     * the best individual = the worst individual
     */
    EvSimplifiedMessyGAOperator selectionMessyGA = 
        new EvSimplifiedMessyGAOperator(1, 2, 0.0, 0.0, 0.0);
    EvPopulation<EvSimplifiedMessyIndividual> selection_population = 
        selectionMessyGA.apply(population);
    
    assertEquals(selection_population.getBestResult(),
        selection_population.getWorstResult());
    assertEquals(selection_population.size(),2);
    assertEquals(selection_population.get(0).getLength(),2);
    assertEquals(selection_population.get(1).getLength(),2);

    /* mutation test
     * in this case mutation_population always should have
     * the best individual value > the worst individual value
     * due to replace gene mutation applied to second individual
     */
    EvSimplifiedMessyGAOperator mutationMessyGA = 
        new EvSimplifiedMessyGAOperator(1, 1, 0.0, 0.0, 1.0);
    EvPopulation<EvSimplifiedMessyIndividual> mutation_population = 
        mutationMessyGA.apply(population);

    assertTrue(
        mutation_population.getBestResult().getObjectiveFunctionValue() >
        mutation_population.getWorstResult().getObjectiveFunctionValue());
    assertEquals(mutation_population.size(),2);
    assertEquals(mutation_population.get(0).getLength(),2);
    assertEquals(mutation_population.get(1).getLength(),2);

    /* general test 1 
     * in this case the new population cannot be worse than old
     * and must have worst individual from old population
     */
    individuals = new EvSimplifiedMessyIndividual[100];
    
    for (int i = 0; i < 100; i++) {
      individuals[i] = new EvSimplifiedMessyIndividual(2);
      individuals[i].addGeneValue(0,1);
      individuals[i].addGeneValue(1,1);
    }

    EvPopulation<EvSimplifiedMessyIndividual> old_population1 =
        new EvPopulation<EvSimplifiedMessyIndividual>(individuals);
    
    old_population1.setObjectiveFunction(messyFn);

    EvSimplifiedMessyGAOperator messyGA = 
        new EvSimplifiedMessyGAOperator(50, 99, 1.0, 1.0, 1.0);
    
    EvPopulation<EvSimplifiedMessyIndividual> new_population1 = 
        messyGA.apply(old_population1);
    
    assertTrue(
        new_population1.getBestResult().getObjectiveFunctionValue() >=
        old_population1.getBestResult().getObjectiveFunctionValue());
    assertTrue(
        new_population1.getWorstResult().getObjectiveFunctionValue() ==
        old_population1.getWorstResult().getObjectiveFunctionValue());
    assertEquals(new_population1.size(),100);
    
    /* general test 2 
     * in this case the new population cannot be better than old
     */
    individuals = new EvSimplifiedMessyIndividual[100];
    
    for (int i = 0; i < 100; i++) {
      individuals[i] = new EvSimplifiedMessyIndividual(2);
      individuals[i].addGeneValue(0,50);
      individuals[i].addGeneValue(1,50);
    }

    EvPopulation<EvSimplifiedMessyIndividual> old_population2 =
        new EvPopulation<EvSimplifiedMessyIndividual>(individuals);
    
    old_population2.setObjectiveFunction(messyFn);

    EvPopulation<EvSimplifiedMessyIndividual> new_population2 = 
        messyGA.apply(old_population2);
    
    assertTrue(
        new_population2.getBestResult().getObjectiveFunctionValue() ==
        old_population2.getBestResult().getObjectiveFunctionValue());
    assertTrue(
        new_population2.getWorstResult().getObjectiveFunctionValue() ==
        old_population2.getWorstResult().getObjectiveFunctionValue());
    assertEquals(new_population2.size(),100);
    
   }
}
