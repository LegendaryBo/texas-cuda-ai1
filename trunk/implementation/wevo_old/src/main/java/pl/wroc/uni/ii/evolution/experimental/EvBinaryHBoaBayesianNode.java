package pl.wroc.uni.ii.evolution.experimental;

import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.bmda.bayesnetwork.EvBinaryBayesianNode;

/**
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvBinaryHBoaBayesianNode extends EvBinaryBayesianNode {

  double max_deviation;
  
  public EvBinaryHBoaBayesianNode(final int i, final double max_deviation_) {
    super(i);
    max_deviation = max_deviation_;
  }
  
  @Override
  public void setProbabilities(double[] probabilities) {
    int count = 0;
    for (int i = 0; i < probabilities.length; i++) {
      for (int j = 0; j < Math.log(probabilities.length); j++) {
        if (inRange(probabilities[i], probabilities[i + (int)Math.pow(2, j)]) ) {
          //System.out.println(i + " is in range of" + (i + Math.pow(2, j)));
          count++;
        }
      }
    }
    System.out.println("uproszczono "+count+" z " + probabilities.length);
  }
  
  private boolean inRange(double a, double b) {
    if (Math.abs(a - b) < max_deviation)
      return true;
    else return false;
  }
  
}
