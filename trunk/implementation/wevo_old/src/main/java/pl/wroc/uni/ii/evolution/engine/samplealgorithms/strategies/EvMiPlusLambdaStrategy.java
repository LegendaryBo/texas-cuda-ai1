package pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvRealVectorWithProbabilitiesIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvReplacementComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.composition.EvTwoOperatorsComposition;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRandomSelection;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilities.EvRealVectorWithProbabilitiesMiLambdaStrategyMutation;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.realvectorwithprobabilities.EvRealVectorWithProbabilitesMiLambdaStrategyCrossover;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvReplacement;

/**
 * Class implements evolutionary algorithm called ES(Mi + Lambda). It uses 4
 * operators: ExtendedTotalRandomSelection, MiLambdaStrategiesRecombination,
 * MiLambdaStrategiesMutation and KBestSelection as replacement.
 * MiLambdaStrategiesRecombination and MiLambdaStrategiesMutation are
 * encapsulated by ChildrenSelector to keep parents population. It uses
 * RealVectorWithProbabilitiesIndividual as individuals.
 * 
 * @author Lukasz Witko, Tomasz Kozakiewicz
 */
public class EvMiPlusLambdaStrategy extends
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
  public EvMiPlusLambdaStrategy(int mi, int lambda, double tau, double tau_prim) {
    super(mi);

    if (mi > lambda) {
      throw new IllegalArgumentException("Lambda must be greater than mi");
    }

    this.mi = mi;
    this.lambda = lambda;
    this.tau = tau;
    this.tau_prim = tau_prim;
  }


  public void init() {

    EvRandomSelection<EvRealVectorWithProbabilitiesIndividual> parent_selection =
        new EvRandomSelection<EvRealVectorWithProbabilitiesIndividual>(lambda,
            true);

    EvRealVectorWithProbabilitiesMiLambdaStrategyMutation mutation_operator =
        new EvRealVectorWithProbabilitiesMiLambdaStrategyMutation(tau, tau_prim);

    EvRealVectorWithProbabilitesMiLambdaStrategyCrossover crossover_operator =
        new EvRealVectorWithProbabilitesMiLambdaStrategyCrossover();

    EvOperator<EvRealVectorWithProbabilitiesIndividual> transformation =
        new EvTwoOperatorsComposition<EvRealVectorWithProbabilitiesIndividual>(
            mutation_operator, crossover_operator);

    EvReplacement<EvRealVectorWithProbabilitiesIndividual> replacement =
        new EvBestFromUnionReplacement<EvRealVectorWithProbabilitiesIndividual>(
            mi);

    EvOperator<EvRealVectorWithProbabilitiesIndividual> all =
        new EvReplacementComposition<EvRealVectorWithProbabilitiesIndividual>(
            new EvTwoOperatorsComposition<EvRealVectorWithProbabilitiesIndividual>(
                transformation, parent_selection), replacement);

    super.addOperatorToEnd(all);
    super.init();

  }
}