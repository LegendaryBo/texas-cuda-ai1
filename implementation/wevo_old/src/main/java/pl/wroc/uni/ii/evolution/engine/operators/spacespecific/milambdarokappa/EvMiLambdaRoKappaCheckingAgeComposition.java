package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.milambdarokappa;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMiLambdaRoKappaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * It can be used for MiLambdaRoKappaIndividual. It removes too old individuals
 * from population. Next it applies a selection operator "op" to population.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */

public class EvMiLambdaRoKappaCheckingAgeComposition implements
    EvOperator<EvMiLambdaRoKappaIndividual> {

  private final int MAX_AGE;

  private EvOperator<EvMiLambdaRoKappaIndividual> op;


  /**
   * @param age max age
   * @param op should be an selection operator
   */
  public EvMiLambdaRoKappaCheckingAgeComposition(int age,
      EvOperator<EvMiLambdaRoKappaIndividual> op) {
    MAX_AGE = age;
    this.op = op;
  }


  public EvPopulation<EvMiLambdaRoKappaIndividual> apply(
      EvPopulation<EvMiLambdaRoKappaIndividual> population) {

    for (int i = population.size() - 1; i >= 0; i--) {
      if (population.get(i).getAge() >= MAX_AGE) {
        population.remove(i);
      }
    }
    population = op.apply(population);

    return population;
  }
}
