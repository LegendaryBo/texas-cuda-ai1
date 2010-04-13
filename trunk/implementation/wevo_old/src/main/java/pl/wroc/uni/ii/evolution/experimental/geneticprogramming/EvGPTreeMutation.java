package pl.wroc.uni.ii.evolution.experimental.geneticprogramming;

import java.util.GregorianCalendar;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.experimental.geneticprogramming.individuals.EvGPTree;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * @author Zbigniew Nazimek, Donata Ma�ecka
 */
public class EvGPTreeMutation implements EvOperator<EvGPTree> {

  EvRandomizer rand =
      new EvRandomizer(new GregorianCalendar().getTimeInMillis());

  double mutation_probability;


  public EvGPTreeMutation(double mutation) {
    mutation_probability = mutation;
  }


  public EvPopulation<EvGPTree> apply(EvPopulation<EvGPTree> population_) {

    EvPopulation<EvGPTree> population =
        (EvPopulation<EvGPTree>) population_.clone();
    EvPopulation<EvGPTree> result = new EvPopulation<EvGPTree>();

    while (population.size() > 0) {
      EvGPTree gpt = population.get(0);
      result.add(processGPTree(gpt));
      population.remove(0);

    }

    return result;
  }


  private EvGPTree processGPTree(EvGPTree gpt) {
    if (rand.nextDouble() < mutation_probability)
      gpt.mutate();
    if (gpt.hasLeft())
      processGPTree(gpt.getLeftSubTree());
    if (gpt.hasRight())
      processGPTree(gpt.getRightSubTree());
    return gpt;
  }

}
