package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * It is reproduction operator. It can be used for MiLambdaRoKappaIndividual. It
 * is a compact of three operators of selection, crossover and mutation. They
 * are set in constructor. It also increase age of each individual in input
 * population.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvMiLambdaRoKappaReproductionComposition implements
    EvOperator<EvMiLambdaRoKappaIndividual> {

  private int children_number;

  private EvOperator<EvMiLambdaRoKappaIndividual> selection_operator;

  private EvOperator<EvMiLambdaRoKappaIndividual> crossover_operator;

  private EvOperator<EvMiLambdaRoKappaIndividual> mutation_operator;

  private boolean children_only;


  /**
   * @param children_number it is number of individuals which this operator will
   *        create and return
   * @param selection_operator it should be a selection operator
   * @param crossover_operator it should be a crossover operator
   * @param mutation_operator it should be a mutation operator
   * @param children_only it determine if parents are taken to output population
   *        of apply method
   */
  public EvMiLambdaRoKappaReproductionComposition(int children_number,
      EvOperator<EvMiLambdaRoKappaIndividual> selection_operator,
      EvOperator<EvMiLambdaRoKappaIndividual> crossover_operator,
      EvOperator<EvMiLambdaRoKappaIndividual> mutation_operator,
      boolean children_only) {

    this.children_number = children_number;
    this.selection_operator = selection_operator;
    this.crossover_operator = crossover_operator;
    this.mutation_operator = mutation_operator;
    this.children_only = children_only;
  }


  public EvPopulation<EvMiLambdaRoKappaIndividual> apply(
      EvPopulation<EvMiLambdaRoKappaIndividual> population) {

    if (population.size() == 0)
      throw new IllegalStateException("Population is empty");

    /* Increasing individuals age. */
    for (int i = 0; i < population.size(); i++) {
      population.get(i).increaseAge();
    }
    EvPopulation<EvMiLambdaRoKappaIndividual> children =
        new EvPopulation<EvMiLambdaRoKappaIndividual>();

    /* Creating children. */
    for (int i = 0; i < children_number; i++) {
      children.addAll(mutation_operator.apply(crossover_operator
          .apply(selection_operator.apply(population))));
    }
    /* _ */
    /* Parents addition. */
    if (!children_only) {
      children.addAll(population);
    }

    return children;
  }

}
