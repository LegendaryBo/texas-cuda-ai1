package pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

public class EvKDeceptiveOneMax implements
    EvObjectiveFunction<EvBinaryVectorIndividual> {

  private static final long serialVersionUID = 1841740375743035854L;

  private int k;


  public EvKDeceptiveOneMax(int k) {
    if (k <= 0) {
      throw new IllegalArgumentException("K parameter to low");
    }

    this.k = k;
  }


  public double evaluate(EvBinaryVectorIndividual individual) {

    int sum = 0;
    int subsum = 0;
    int subseq = 0;

    int dimension = individual.getDimension();

    for (int i = 0; i < dimension; i++) {
      if (individual.getGene(i) == 1)
        subsum += 1;

      subseq++;

      if (subseq == k) {
        subseq = 0;
        sum += (subsum == 0 ? (k + 1) : subsum);
        subsum = 0;
      }
    }

    sum += subsum;

    return (double) sum;

  }

}
