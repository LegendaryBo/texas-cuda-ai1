package pl.wroc.uni.ii.evolution.grammar;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvKnaryIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryIndividualMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.EvKnaryVectorOnePointCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.conditions.EvTerminationCondition;
import pl.wroc.uni.ii.evolution.solutionspaces.EvKnaryVectorSpace;

/**
 * @author Konrad Drukala (heglion@gmail.com)
 * @author Marta Stañska (martastanska@gmail.com) This class SHOULD extend
 *         EvAlgorithm class, but there is no time for it now.
 */
public class EvGrammarEvolution {
  private int numberOfIndividuals;

  private int codonSize;

  private int numberOfCodons;

  private EvObjectiveFunction<EvKnaryIndividual> objectiveFunction;

  private int indvidualMaxValue;

  private double mutationProbability;

  private EvTerminationCondition<EvKnaryIndividual> terminationCondition;


  /**
   * Default constructor
   * 
   * @param numberOfIndividuals How many individuals we want
   * @param codonSize how many genes are used per codon
   * @param numberOfCodons number of codons
   * @param objectiveFunction objective function for evaluation
   * @param terminationCondition termination condition for checking whether to
   *        finish evolution
   */
  public EvGrammarEvolution(int numberOfIndividuals, int codonSize,
      int numberOfCodons,
      EvObjectiveFunction<EvKnaryIndividual> objectiveFunction,
      EvTerminationCondition<EvKnaryIndividual> terminationCondition) {
    this(numberOfIndividuals, codonSize, numberOfCodons, objectiveFunction, 2,
        terminationCondition);
  }


  /**
   * Default constructor
   * 
   * @param numberOfIndividuals How many individuals we want
   * @param codonSize how many genes are used per codon
   * @param numberOfCodons number of codons
   * @param objectiveFunction objective function for evaluation
   * @param individualMaxValue determines maximum value of gene
   * @param terminationCondition termination condition for checking whether to
   *        finish evolution
   */
  public EvGrammarEvolution(int numberOfIndividuals, int codonSize,
      int numberOfCodons,
      EvObjectiveFunction<EvKnaryIndividual> objectiveFunction,
      int individualMaxValue,
      EvTerminationCondition<EvKnaryIndividual> terminationCondition) {
    this.numberOfIndividuals = numberOfIndividuals;
    this.codonSize = codonSize;
    this.numberOfCodons = numberOfCodons;
    this.objectiveFunction = objectiveFunction;
    this.indvidualMaxValue = individualMaxValue;
    this.mutationProbability = 0.02;
    this.terminationCondition = terminationCondition;
  }


  /**
   * Runs grammar evolution
   * 
   * @return best individual
   */
  public EvKnaryIndividual run() {
    EvKnaryVectorSpace solutionSpace =
        new EvKnaryVectorSpace(this.codonSize * this.numberOfCodons,
            this.indvidualMaxValue);
    solutionSpace.setObjectiveFuntion(this.objectiveFunction);
    EvAlgorithm<EvKnaryIndividual> alg =
        new EvAlgorithm<EvKnaryIndividual>(this.numberOfIndividuals);
    alg.setSolutionSpace(solutionSpace);
    alg.setTerminationCondition(this.terminationCondition);
    alg.setObjectiveFunction(this.objectiveFunction);

    /*
     * "It has been found that the standard one point crossover operator already
     * employed in the algorithm is the most consistent of those examined, in
     * general producing more successful runs." Crossover in Grammatical
     * Evolution: A Smooth Operator? -Michael O'Neill and Conor Ryan
     */
    EvKnaryVectorOnePointCrossover<EvKnaryIndividual> crossover =
        new EvKnaryVectorOnePointCrossover<EvKnaryIndividual>();
    alg.addOperator(crossover);

    // Mutation should be on the bits
    alg.addOperator(new EvKnaryIndividualMutation(this.mutationProbability));

    alg.init();
    alg.run();

    return alg.getBestResult();
  }
}