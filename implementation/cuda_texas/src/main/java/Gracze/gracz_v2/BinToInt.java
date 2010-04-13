package Gracze.gracz_v2;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

public class BinToInt {

  public static int decode(EvBinaryVectorIndividual individual, int start_position, int length) {
    
    int power = 1;
    int sum = 0;
    for (int i=0; i < length; i++) {
      if (individual.getGene(start_position + i) == 1)
        sum += power;
      power = power * 2;
    }
    
    return sum;
  }
  
}
