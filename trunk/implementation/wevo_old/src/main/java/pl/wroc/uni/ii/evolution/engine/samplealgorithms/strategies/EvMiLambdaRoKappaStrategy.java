package pl.wroc.uni.ii.evolution.engine.samplealgorithms.strategies;

import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRouletteSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvNearZeroFitness;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa.EvMiLambdaRoKappaGlobalIntermediaryRecombination;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa.EvMiLambdaRoKappaCheckingAgeComposition;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa.EvMiLambdaRoKappaReproductionComposition;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa.EvMiLambdaRoKappaRotationMutation;

/**
 * Class implements evolutionary algorithm called ES(Mi, Lambda, Ro, Kappa) and
 * ES(Mi + Lambda, Ro, Kappa). It uses 2 operators:
 * ReproductionForMiLambdaRoKappaStrategy, ReplacementWithCheckingAge.
 * ReplacementWithCheckingAge encapsulate KBestSelection operator and
 * ReproductionForMiLambdaRoKappaStrategy encapsulate 3 operators:
 * RouletteSelection, GlobalIntermediaryRecombination and RotationMutation. It
 * could keep parents population, but it doesn't have to. It uses
 * MiLambdaRoKappaIndividual as individuals.
 * 
 * @author Lukasz Witko, Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvMiLambdaRoKappaStrategy extends
    EvAlgorithm<EvMiLambdaRoKappaIndividual> {

  private int mi;

  private int lambda;

  private int ro;

  private int max_age;

  private double tau;

  private double tau_prim;

  private double beta;

  private boolean children_only;


  /**
   * @param mi number of individuals in population
   * @param lambda number of individuals in children population
   * @param ro how many parement will be used to create a child
   * @param tau a parameter for alfa vector mutation
   * @param tau_prim a parameter for alfa vector mutation
   * @param beta a parameter for beta vector mutation (recomended value: 0.0873)
   * @param max_age how many iteration individual can survive
   * @param children_only
   */

  public EvMiLambdaRoKappaStrategy(int mi, int lambda, int ro, double tau,
      double tau_prim, double beta, int max_age, boolean children_only) {
    super(mi);

    this.mi = mi;
    this.lambda = lambda;
    this.ro = ro;
    this.tau = tau;
    this.tau_prim = tau_prim;
    this.beta = beta;
    this.max_age = max_age;
    this.children_only = children_only;

    if (mi > lambda) {
      throw new IllegalArgumentException("Lambda must be greater than mi");
    }
  }


  @Override
  public void init() {

    EvRouletteSelection<EvMiLambdaRoKappaIndividual> selection_operator =
        new EvRouletteSelection<EvMiLambdaRoKappaIndividual>(
            new EvNearZeroFitness<EvMiLambdaRoKappaIndividual>(), ro);

    EvMiLambdaRoKappaGlobalIntermediaryRecombination recombination_operator =
        new EvMiLambdaRoKappaGlobalIntermediaryRecombination();

    EvMiLambdaRoKappaRotationMutation mutation_operator =
        new EvMiLambdaRoKappaRotationMutation(beta, tau, tau_prim);

    EvMiLambdaRoKappaReproductionComposition reproduction_operator =
        new EvMiLambdaRoKappaReproductionComposition(lambda,
            selection_operator, recombination_operator, mutation_operator,
            children_only);

    EvKBestSelection<EvMiLambdaRoKappaIndividual> kbest_operator =
        new EvKBestSelection<EvMiLambdaRoKappaIndividual>(mi);

    EvMiLambdaRoKappaCheckingAgeComposition replacement_operator =
        new EvMiLambdaRoKappaCheckingAgeComposition(max_age, kbest_operator);

    addOperatorToEnd(reproduction_operator);
    addOperatorToEnd(replacement_operator);

    super.init();

  }
}