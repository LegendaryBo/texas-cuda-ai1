package generator;

import java.util.Random;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;


/**
 * 
 * Prosty generator zupelnei losowych graczy
 * 
 * 
 * @author Kacper Gorski (railman@gmail.com)
 *
 */
public class IndividualGenerator {

  private Random generator = null;
  private int size = 0;
  
  public IndividualGenerator(int seed, int size_) {
    generator = new Random(seed);
    size = size_;
  }
  
  public EvBinaryVectorIndividual generate() {
   
    int[] genes = new int[1 + size/32];
      
    for (int i=0; i < 1 + size/32; i++)
      genes[i] = generator.nextInt();
    
    return new EvBinaryVectorIndividual(genes,size);
    
  }

  public void reset() {
    // TODO Auto-generated method stub
    throw new IllegalStateException("kurwa");
    
    
  }
  
}
