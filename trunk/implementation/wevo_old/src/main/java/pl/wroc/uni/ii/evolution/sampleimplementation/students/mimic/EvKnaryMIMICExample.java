package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;

/**
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 *
 */
public final class EvKnaryMIMICExample {
  
  
  /**
   * There is no need to use it.
   */
  private EvKnaryMIMICExample() {
  }
  
  
  /**
   * Example of the MIMIC algorithm.
   * @param args none
   */
  public static void main(final String[] args) {
  
    // off MagicNumber
    int dimension = 10; // dimension of each individual
    int number_of_values = 5;
    int size = 100; // size of population
    int size_stay = 50; // size of population after selection
    int iterations = 10; // number of iterations
    // on MagicNumber
    
    EvAlgorithm<EvKnaryIndividual> mimic;
    mimic = new EvAlgorithm<EvKnaryIndividual>(size);
    
    // termination condition
    mimic.setTerminationCondition(new 
        EvMaxIteration<EvKnaryIndividual>(iterations));
    
    // objective function
    EvKnarySomeObjectiveFunction objective_function = 
        new EvKnarySomeObjectiveFunction();
    mimic.setObjectiveFunction(objective_function);
      
    // 1. operator (selection)
    EvKBestSelection<EvKnaryIndividual> selection;
    selection = new EvKBestSelection<EvKnaryIndividual>(size_stay);
    mimic.addOperatorToEnd(selection);
    
    // 2. operator (MIMIC)
    EvKnaryMIMICOperator mimic_operator; 
    mimic_operator = new EvKnaryMIMICOperator(dimension, size, 
        number_of_values, objective_function);
    mimic.addOperatorToEnd(mimic_operator);
    
    // solution space
    mimic.setSolutionSpace(mimic_operator.getSolutionSpace());
    
    mimic.init();
    
    mimic.run();
    
    EvKnaryIndividual mimic_best = mimic.getBestResult();
    double mimic_best_value = mimic_best.getObjectiveFunctionValue();
    
    EvKnaryIndividual mimic_worst = mimic.getWorstResult();
    double mimic_worst_value = mimic_worst.getObjectiveFunctionValue();
    
    System.out.println("MIMIC - result of the example:");
    System.out.println("max possible value = 8");
    System.out.println("best:  " + mimic_best_value + " - " + mimic_best);
    System.out.println("worst: " + mimic_worst_value + " - " + mimic_worst);
  }
}





/**
 * Example objective function.
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 */
class EvKnarySomeObjectiveFunction implements 
  EvObjectiveFunction<EvKnaryIndividual> {

  

  
  /**
   * {@inheritDoc}
   */
  public double evaluate(final EvKnaryIndividual individual) {
    int sum = 0;
    for (int i = 1; i < individual.getDimension(); i++) {
      if (individual.getGene(i) > individual.getGene(i - 1)) {
        sum++;
      }
    }
    return sum;
  }
}
