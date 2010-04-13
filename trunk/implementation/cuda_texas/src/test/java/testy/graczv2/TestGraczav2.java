package testy.graczv2;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import Gracze.GraczAIv2;

public class TestGraczav2 extends TestCase {

  public void testGracz() {
    
    EvBinaryVectorIndividual individual = new EvBinaryVectorIndividual(108);
    
    GraczAIv2 gracz = new GraczAIv2(individual, 0);
    

    
  }
  
}
