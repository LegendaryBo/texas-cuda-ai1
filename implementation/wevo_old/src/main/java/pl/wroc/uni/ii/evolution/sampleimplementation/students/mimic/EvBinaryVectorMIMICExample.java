/**
 * 
 */
package pl.wroc.uni.ii.evolution.sampleimplementation.students.mimic;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;

/**
 * Example of the MIMIC algorithm - binary version.
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 */
public final class EvBinaryVectorMIMICExample {

  
  /**
   * There is no need to use it.
   */
  private EvBinaryVectorMIMICExample() {
  }
  
  
  /**
   * Example of the MIMIC algorithm.
   * @param args none
   */
  public static void main(final String[] args) {
  
    // off MagicNumber
    int dimension = 10; // dimension of each individual
    int size = 100; // size of population
    int size_stay = 50; // size of population after selection
    int iterations = 10; // number of iterations
    // on MagicNumber
    
    EvAlgorithm<EvBinaryVectorIndividual> mimic;
    mimic = new EvAlgorithm<EvBinaryVectorIndividual>(size);
    
    // termination condition
    mimic.setTerminationCondition(new 
        EvMaxIteration<EvBinaryVectorIndividual>(iterations));
    
    // objective function
    EvBinaryVectorSomeObjectiveFunction objective_function = 
        new EvBinaryVectorSomeObjectiveFunction();
    mimic.setObjectiveFunction(objective_function);
      
    // 1. operator (selection)
    EvKBestSelection<EvBinaryVectorIndividual> selection;
    selection = new EvKBestSelection<EvBinaryVectorIndividual>(size_stay);
    mimic.addOperatorToEnd(selection);
    
    // 2. operator (MIMIC)
    EvBinaryVectorMIMICOperator mimic_operator; 
    mimic_operator = new EvBinaryVectorMIMICOperator(dimension, size, 
        objective_function);
    mimic.addOperatorToEnd(mimic_operator);
    
    // solution space
    mimic.setSolutionSpace(mimic_operator.getSolutionSpace());
    
    mimic.init();
    
    mimic.run();
    
    EvBinaryVectorIndividual mimic_best = mimic.getBestResult();
    double mimic_best_value = mimic_best.getObjectiveFunctionValue();
    
    EvBinaryVectorIndividual mimic_worst = mimic.getWorstResult();
    double mimic_worst_value = mimic_worst.getObjectiveFunctionValue();
    
    System.out.println("MIMIC - result of the example:");
    System.out.println("best value = 9");
    System.out.println("best:  " + mimic_best_value + " - " + mimic_best);
    System.out.println("worst: " + mimic_worst_value + " - " + mimic_worst);
  }
}





/**
 * Example objective function.
 * @author Sabina Fabiszewska (sabina.fabiszewska@gmail.com)
 */
class EvBinaryVectorSomeObjectiveFunction implements 
  EvObjectiveFunction<EvBinaryVectorIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = -909916286044841394L;

  
  /**
   * {@inheritDoc}
   */
  public double evaluate(final EvBinaryVectorIndividual individual) {
    int sum = 0;
    for (int i = 1; i < individual.getDimension(); i++) {
      if (individual.getGene(i) != individual.getGene(i - 1)) {
        sum++;
      }
    }
    return sum;
  }
}