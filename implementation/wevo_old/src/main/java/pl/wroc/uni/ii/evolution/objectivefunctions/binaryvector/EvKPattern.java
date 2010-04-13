package pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

public class EvKPattern implements
    EvObjectiveFunction<EvBinaryVectorIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = -6626453602510086675L;

  private int reward = 0;

  private int[] pattern = null;


  public EvKPattern(int[] pattern, int reward) {
    this.reward = reward;
    this.pattern = pattern;
  }


  public double evaluate(EvBinaryVectorIndividual individual) {

    int pos = 0;
    double val = 0.0;
    while (pos < individual.getDimension()) {

      int count = countMatched(pos, individual, pattern);
      if (count == pattern.length) {
        val += reward;
      }

      val += countOnce(pos, individual);

      pos += pattern.length;
    }

    return val;
  }


  private double countOnce(int pos, EvBinaryVectorIndividual individual) {

    int count = 0;

    for (int i = pos; (i < pos + pattern.length)
        && (i < individual.getDimension()); i++) {
      if (individual.getGene(i) == 1) {
        count += 1;
      }
    }
    return count;

  }


  private int countMatched(int pos, EvBinaryVectorIndividual ind, int[] pattern) {

    int matched = 0;

    for (int i = pos; (i < pos + pattern.length) && (i < ind.getDimension()); i++) {
      if (pattern[i - pos] == ind.getGene(i)) {
        matched += 1;
      }
    }
    return matched;
  }
}
