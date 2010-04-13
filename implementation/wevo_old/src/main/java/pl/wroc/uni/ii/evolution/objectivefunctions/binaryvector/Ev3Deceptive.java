package pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

public class Ev3Deceptive implements
    EvObjectiveFunction<EvBinaryVectorIndividual> {

  /**
   * 
   */
  private static final long serialVersionUID = 429800845815902201L;


  public double evaluate(EvBinaryVectorIndividual individual) {

    float f = 0;
    int s = 0;
    int i = 0;

    for (i = 0; i < individual.getDimension();) {

      s = (individual.getGene(i++) == 1) ? 1 : 0;
      s += (individual.getGene(i++) == 1) ? 1 : 0;
      s += (individual.getGene(i++) == 1) ? 1 : 0;

      if (s == 0) {
        f += 0.9;
      } else {
        if (s == 1) {
          f += 0.8;
        } else {
          if (s == 3) {
            f += 1;
          }
        }
      }
    }
    return f;
  }

}
