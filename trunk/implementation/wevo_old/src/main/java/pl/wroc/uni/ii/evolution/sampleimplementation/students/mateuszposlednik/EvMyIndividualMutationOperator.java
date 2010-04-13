package pl.wroc.uni.ii.evolution.sampleimplementation.students.mateuszposlednik;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * Operator mutujacy dla EvMyIndividual (zadanie rozgrzewkowe). Losuje
 * 0..wielkosc populacji osobnikow Zamienia w kazdym z nich dwa geny miejscami.
 * 
 * @author Mateusz Poslednik mateusz.poslednik@gmail.com
 */
public class EvMyIndividualMutationOperator implements
    EvOperator<EvMyIndividual> {

  /**
   * Zwraca nowa populacje po mutacji.
   * 
   * @param population Populacja w danej iteracji
   * @return Nowa populacja
   */
  public EvPopulation<EvMyIndividual> apply(
      final EvPopulation<EvMyIndividual> population) {

    EvObjectiveFunction<EvMyIndividual> obj = null;
    int population_size = population.size();
    int iterations = EvRandomizer.INSTANCE.nextInt(0, population_size);

    if (population_size > 0) {
      obj = population.get(0).getObjectiveFunction();
    }

    for (int i = 0; i < iterations; i++) {
      int number_individual = EvRandomizer.INSTANCE.nextInt(0, population_size);

      EvMyIndividual in = population.get(number_individual);
      int[] genes = in.getGenes();
      int gen1 = EvRandomizer.INSTANCE.nextInt(0, genes.length);
      int gen2 = EvRandomizer.INSTANCE.nextInt(0, genes.length);
      int temp = genes[gen1];
      genes[gen1] = genes[gen2];
      genes[gen2] = temp;
      EvMyIndividual in_mutated = new EvMyIndividual(genes);
      in_mutated.setObjectiveFunction(obj);
      population.set(number_individual, in_mutated);
    }
    return population;
  }

}
