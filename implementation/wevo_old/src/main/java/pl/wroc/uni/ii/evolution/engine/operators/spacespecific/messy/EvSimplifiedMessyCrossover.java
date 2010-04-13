package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import java.util.ArrayList;
import java.util.List;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.operators.EvCrossover;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

/**
 * A standard crossover for EvMessyIndividual.
 * 
 * @author Marcin Golebiowski
 */
public class EvSimplifiedMessyCrossover extends
    EvCrossover<EvSimplifiedMessyIndividual> {

  private double crossover_probability;


  /**
   * @param crossover_probability probability of crossver
   */
  public EvSimplifiedMessyCrossover(double crossover_probability) {
    this.crossover_probability = crossover_probability;
  }


  @Override
  public int arity() {
    return 2;
  }


  @Override
  public List<EvSimplifiedMessyIndividual> combine(
      List<EvSimplifiedMessyIndividual> parents) {
    assert parents.size() == 2;

    List<EvSimplifiedMessyIndividual> result =
        new ArrayList<EvSimplifiedMessyIndividual>();

    EvSimplifiedMessyIndividual father = parents.get(0);
    EvSimplifiedMessyIndividual mother = parents.get(1);

    if (!EvRandomizer.INSTANCE.nextProbableBoolean(crossover_probability)) {
      result.add(father);
      result.add(mother);
    } else {

      EvSimplifiedMessyIndividual daughter =
          new EvSimplifiedMessyIndividual(father.getLength());
      EvSimplifiedMessyIndividual son =
          new EvSimplifiedMessyIndividual(father.getLength());

      for (int position = 0; position < father.getLength(); position++) {

        ArrayList<Integer> father_and_mother_gene_values =
            new ArrayList<Integer>();

        if (father.getGeneValues(position) != null) {
          father_and_mother_gene_values.addAll(father.getGeneValues(position));
        }
        if (mother.getGeneValues(position) != null) {
          father_and_mother_gene_values.addAll(mother.getGeneValues(position));
        }

        // for each gene value randomly choose if vale go to son or daugter
        for (int gene_value : father_and_mother_gene_values) {
          if (EvRandomizer.INSTANCE.nextProbableBoolean(0.5)) {
            son.addGeneValue(position, gene_value);
          } else {
            daughter.addGeneValue(position, gene_value);
          }

        }
      }

      son.removeAllDuplicateGeneValues();
      son.setObjectiveFunction(father.getObjectiveFunction());
      daughter.removeAllDuplicateGeneValues();
      daughter.setObjectiveFunction(father.getObjectiveFunction());

      result.add(son);
      result.add(daughter);

    }
    return result;
  }


  @Override
  public int combineResultSize() {
    return 2;
  }

}
