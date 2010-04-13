package pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilities.EvRealVectorWithProbabilitiesMiLambdaStrategyMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilities.EvRealVectorWithProbabilitesMiLambdaStrategyCrossover;

/**
 * Class implements evolutionary algorithm called ES(Mi, Lambda). It uses 4
 * operators: TotalRandomSelection, MiLambdaStrategiesRecombination,
 * MiLambdaStrategiesMutation and KBestSelection as replacement. It uses
 * RealVectorWithProbabilitiesIndividual as individuals.
 * 
 * @author Lukasz Witko, Tomasz Kozakiewicz
 */
public class EvMiLambdaStrategy extends
    EvAlgorithm<EvRealVectorWithProbabilitiesIndividual> {

  private int mi;

  private int lambda;

  private double tau;

  private double tau_prim;


  /**
   * @param mi number of individuals in population
   * @param lambda number of individuals in children population
   * @param tau a parameter of mutation
   * @param tau_prim a parameter of mutation
   */
  public EvMiLambdaStrategy(int mi, int lambda, double tau, double tau_prim) {
    super(mi);

    if (mi > lambda) {
      throw new IllegalArgumentException("Lambda must be greater than mi");
    }

    this.mi = mi;
    this.lambda = lambda;
    this.tau = tau;
    this.tau_prim = tau_prim;
  }


  @Override
  public void init() {

    EvRandomSelection<EvRealVectorWithProbabilitiesIndividual> reproduction_operator =
        new EvRandomSelection<EvRealVectorWithProbabilitiesIndividual>(lambda,
            true);

    EvRealVectorWithProbabilitesMiLambdaStrategyCrossover crossover_operator =
        new EvRealVectorWithProbabilitesMiLambdaStrategyCrossover();

    EvRealVectorWithProbabilitiesMiLambdaStrategyMutation mutation_operator =
        new EvRealVectorWithProbabilitiesMiLambdaStrategyMutation(tau, tau_prim);

    EvKBestSelection<EvRealVectorWithProbabilitiesIndividual> selection_operator =
        new EvKBestSelection<EvRealVectorWithProbabilitiesIndividual>(mi);

    super.addOperatorToEnd(reproduction_operator);
    super.addOperatorToEnd(crossover_operator);
    super.addOperatorToEnd(mutation_operator);
    super.addOperatorToEnd(selection_operator);
    super.init();

  }
}