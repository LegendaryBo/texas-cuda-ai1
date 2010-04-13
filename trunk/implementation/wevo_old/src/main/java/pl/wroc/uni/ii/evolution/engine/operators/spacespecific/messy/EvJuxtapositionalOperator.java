package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * The Messy Juxtapositional Phase Operator. This operator implements second
 * phase of Messy Genetic Algorithm. Consecutively applies three operators:
 * MessyCutSpliceOperator, MessyBinaryVectorNegationMutation and
 * MessyGenicMutation.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvJuxtapositionalOperator implements
    EvOperator<EvMessyBinaryVectorIndividual> {

  // Operators used by the Juxtapositional
  protected EvMessyPairsTournamentSelection<EvMessyBinaryVectorIndividual> selection;

  protected EvMessyCutSpliceOperator<EvMessyBinaryVectorIndividual> cutsplice;

  protected EvMessyBinaryVectorNegationMutation allelic_mutation;

  protected EvMessyGenicMutation<EvMessyBinaryVectorIndividual> genic_mutation;


  /**
   * Constructor, creates the Juxtapositional operator.
   * 
   * @param cut_probability cut probability in CutSplice operator
   * @param splice_probability splice probability in CutSplice operator
   * @param allelic_mutation_probability mutation probability in
   *        BinaryVectorNegationMutation
   * @param genic_mutation_probability, mutation probability in GenicMutation
   * @param tie_breaking tie breaking enabled flag
   * @param thresholding thresholding enabled flag
   */
  public EvJuxtapositionalOperator(double cut_probability,
      double splice_probability, double allelic_mutation_probability,
      double genic_mutation_probability, boolean thresholding,
      boolean tie_breaking) {

    selection =
        new EvMessyPairsTournamentSelection<EvMessyBinaryVectorIndividual>(1,
            thresholding, tie_breaking);

    cutsplice =
        new EvMessyCutSpliceOperator<EvMessyBinaryVectorIndividual>(
            cut_probability, splice_probability);
    cutsplice.setCombineParentSelector(selection);

    allelic_mutation =
        new EvMessyBinaryVectorNegationMutation(allelic_mutation_probability);

    genic_mutation =
        new EvMessyGenicMutation<EvMessyBinaryVectorIndividual>(
            genic_mutation_probability);
  }


  /**
   * Consecutively applies to the population three operators: MessyCutSplice,
   * NegationMutation and GenicMutation.
   * 
   * @param population input population
   * @return population population after one iteration of Juxtapositional
   */
  public EvPopulation<EvMessyBinaryVectorIndividual> apply(
      EvPopulation<EvMessyBinaryVectorIndividual> population) {

    population = cutsplice.apply(population);
    population = allelic_mutation.apply(population);
    population = genic_mutation.apply(population);

    return population;
  }

}