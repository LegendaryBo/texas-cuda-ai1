package pl.wroc.uni.ii.evolution.engine.prototype;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

/**
 * Returns a population of binary individuals so you need only
 * to override other methods. It is good only for operators that
 * can operate on BinaryIndividuals
 *
 */
public abstract class EvOperatorWithBinaryIndividuals extends EvOperatorTestCase<EvBinaryVectorIndividual> {

  @SuppressWarnings("unchecked")
  @Override
  protected EvPopulation populationWithObjectiveFunctionSet() {
    EvPopulation<EvBinaryVectorIndividual> population = 
      new EvPopulation<EvBinaryVectorIndividual>();
    
    for (int i = 0; i < 10; i++) 
      population.add(new EvBinaryVectorIndividual(new int[] {1, 0}));
    
    population.setObjectiveFunction(new EvOneMax());
    
    return population;
  }

}