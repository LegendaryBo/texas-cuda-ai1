package pl.wroc.uni.ii.evolution.sampleimplementation;

import java.util.Arrays;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

public class EvSGAAppletObjectiveFunction implements
    EvObjectiveFunction<EvBinaryVectorIndividual> {

  private static final long serialVersionUID = -3823464549485153748L;

  private double[] weights;

  private double[] prices;

  private double limit = 0;

  private int CHROM_LEN;


  public EvSGAAppletObjectiveFunction(int cl) {
    CHROM_LEN = cl;
    weights = new double[CHROM_LEN];
    prices = new double[CHROM_LEN];
    for (int i = 0; i < CHROM_LEN; i++) {
      weights[i] = (double) Math.round((Math.random() * CHROM_LEN + 1));
      prices[i] = (double) Math.round((Math.random() * 500 + 1));
    }
    double[] x = weights.clone();
    Arrays.sort(x);
    for (int i = 0; i < 15; i++)
      limit += x[i];
  }


  public double getSumPrices() {
    double r = 0.0;
    for (int i = 0; i < CHROM_LEN; i++) {
      r += prices[i];
    }
    return r;
  }


  public double evaluate(EvBinaryVectorIndividual individual) {
    double w = 0.0, pr = 0.0;
    for (int i = 0; i < CHROM_LEN; i++) {
      if (individual.getGene(i) == 1) {
        w += weights[i];
        pr += prices[i];
      }
    }
    if (w < limit)
      return pr;
    else
      return 0.6 * pr * (limit / w);
  }


  public String toString() {
    return toString(null);
  }


  public String toString(EvBinaryVectorIndividual ind) {
    String s = "<table><tr>";
    for (int i = 0; i < CHROM_LEN; i++) {
      s += "<td><small>";
      if (ind != null && (ind.getGene(i) == 1)) {
        s +=
            "<font color=\"#ff0000\">(" + prices[i] + "," + weights[i]
                + ")</font> ";
      } else {
        s += "(" + prices[i] + "," + weights[i] + ") ";
      }
      s += "</small></td>";
      if (i % 5 == 0 && i > 0) {
        s += "</tr><tr>";
      }
    }
    s += "</tr></table>";
    s += " weight limit: " + limit;
    if (ind != null)
      s += " solution quality: " + ind.getObjectiveFunctionValue();
    if (ind != null) {
      double weight = 0.0, value = 0.0;
      for (int i = 0; i < CHROM_LEN; i++) {
        if (ind.getGene(i) == 1) {
          weight += weights[i];
          value += prices[i];
        }
      }
      s += "<br/> weight: " + weight + " value: " + value;
    }
    return s;
  }
}
