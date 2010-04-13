package pl.wroc.uni.ii.evolution.experimental.masterslave;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * A simple example of master-slave distribution model. 
 * This is a master component.
 * 
 * @author Karol 'Asgaroth' Stosiek (karol.stosiek@gmail.com)
 * @author Mateusz 'm4linka' Malinowski (m4linka@gmail.com)
 */
public final class EvMasterSlaveMasterExample {
  
  /**
   * Main routine. 
   * 
   * @param args - ignored
   */
  public static void main(final String[] args) {
    
    /*
     * Choosing an objective function.
     */
    EvOneMax objectiveFunction = new EvOneMax();
    
    /*
     * Creating algorithm instace with population size
     * set to 100 individuals.
     */
    
    final int populationSize = 100;
    EvAlgorithm<EvBinaryVectorIndividual> genericEA =
        new EvAlgorithm<EvBinaryVectorIndividual>(populationSize);
    
    /*
     * Setting up solution space, with chosen objective function
     * and chromosome length fixed to 30. 
     */
    
    final int chromosomeLength = 30;
    EvBinaryVectorSpace solutionSpace = 
      new EvBinaryVectorSpace(objectiveFunction, chromosomeLength);
   
    /*
     * Choosing selection method. We want to use KBestSelection operator
     * and preserve 50 individuals in population with each application. 
     */
    
    final int individualsPreserved = 50;
    EvKBestSelection<EvBinaryVectorIndividual> selectionMethod = 
      new EvKBestSelection<EvBinaryVectorIndividual>(individualsPreserved);
    
    /*
     * In this part we choose, which operators to add to our
     * algorithm. First of all, we choose a crossover method.
     */
    EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual> crossoverMethod =
      new EvKnaryVectorUniformCrossover<EvBinaryVectorIndividual>();
    
    final int iterations = 100;
    
    EvMaxIteration<EvBinaryVectorIndividual> terminationCondition =
      new EvMaxIteration<EvBinaryVectorIndividual>(iterations);
    
    /*
     * We set the solution space, objective function 
     * and add operators to our algorithm.  
     */
    genericEA.setSolutionSpace(solutionSpace);
    genericEA.setObjectiveFunction(objectiveFunction);
    genericEA.addOperatorToEnd(selectionMethod);
    genericEA.addOperatorToEnd(crossoverMethod);
    genericEA.setTerminationCondition(terminationCondition);

    /**********************************************************************/
    
    EvPopulationDistributor<EvBinaryVectorIndividual> population_distributor =
        new EvEqualPopulationDistributor<EvBinaryVectorIndividual>();
    
    EvDBConnector db_connector = new EvDBConnector();
    
    EvMaster<EvBinaryVectorIndividual> master = new EvMaster(
        genericEA, population_distributor);
    
    // optional method
    master.setDBConnector(db_connector);
    
    master.run();

    /**********************************************************************/
    
    /*
     * We create new evolutionary task, set the algorithm
     * to the one we have just specified and run the evolution,
     * awaiting the best result to be printed to the standard output
     * after the termination condition is reached.
     */
    EvTask evolutionaryTask = new EvTask();
   
    evolutionaryTask.setAlgorithm(genericEA);
    evolutionaryTask.run();
    evolutionaryTask.printBestResult();
  }
  
  /**
   * Empty constructor to satisfy style checking rules.
   */
  private EvMasterSlaveMasterExample() {
    
  }
  
}
