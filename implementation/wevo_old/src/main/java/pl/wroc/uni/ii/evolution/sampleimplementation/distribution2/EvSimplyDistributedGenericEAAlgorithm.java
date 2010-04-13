package pl.wroc.uni.ii.evolution.sampleimplementation.distribution2;

import java.net.MalformedURLException;
import java.net.URL;

import pl.wroc.uni.ii.evolution.distribution2.operators.EvSimpleDistributionGetOperator;
import pl.wroc.uni.ii.evolution.distribution2.operators.EvSimpleDistributionSendOperator;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorUniformCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

/**
 * Simple example of an algorithm distributed with 
 * simple distribution package. 
 * 
 * @author Karol Stosiek (karol.stosiek@gmail.com)
 */
public final class EvSimplyDistributedGenericEAAlgorithm {

  /**
   * This construtor disables the generation of public 
   * default constructor in java class file.
   */
  private EvSimplyDistributedGenericEAAlgorithm() {
  }
  
  /**
   * Static method to run the algorithm.
   * 
   * @param args - command line arguments. Ignored.
   * @throws MalformedURLException - thrown, when the URL of the server
   *                                 is malformed. 
   */
  public static void main(final String[] args) throws MalformedURLException {

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
    
    /*
     * Now we will set up operators that provide distribution.
     * We have to create an URL to the distribution server. 
     */
    URL serverURL = new URL("http://localhost:8080/");

    /*
     * We create the operators, both for sending and receiving individuals,
     * with the specified URL address and fixed amount of individuals 
     * to send to/receive from the server.
     */
    
    final int mobileIndividuals = 10;
    
    EvSimpleDistributionSendOperator<EvBinaryVectorIndividual> sendOperator =
      new EvSimpleDistributionSendOperator<EvBinaryVectorIndividual>(
          serverURL, mobileIndividuals);
    
    EvSimpleDistributionGetOperator<EvBinaryVectorIndividual> getOperator =
      new EvSimpleDistributionGetOperator<EvBinaryVectorIndividual>(
          serverURL, mobileIndividuals);
    
    /*
     * Finally, we have to set up the termination condition for our algorithm.
     */
    
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
    genericEA.addOperatorToEnd(sendOperator);
    genericEA.addOperatorToEnd(getOperator);
    genericEA.setTerminationCondition(terminationCondition);

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
}
