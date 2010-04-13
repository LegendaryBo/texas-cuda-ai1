package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;
/**
 * @author: Piotr Baraniak, Marek Chrusciel 
 */

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.utils.EvRandomizer;

public class EvMutationForBinaryIndividualsTest extends TestCase {
  private int INDIV_DIMENSION = 100;
  private EvBinaryVectorIndividual test;
  private EvBinaryVectorIndividual pattern;
  
  protected void setUp() throws Exception {
    super.setUp();
    test = new EvBinaryVectorIndividual(INDIV_DIMENSION);
   
    for(int i = 0; i < INDIV_DIMENSION; i++) {
      test.setGene(i, EvRandomizer.INSTANCE.nextInt(2));
    }
    
     pattern = test.clone();
  }
  
 
  public void testMutationProbabilityZero(){
    EvBinaryVectorNegationMutation operator = new EvBinaryVectorNegationMutation(0.0);
    EvPopulation<EvBinaryVectorIndividual> tmp = new EvPopulation<EvBinaryVectorIndividual>();
    tmp.add( test );
    EvPopulation<EvBinaryVectorIndividual> result;
    result = operator.apply( tmp );
    for(int i = 0; i < INDIV_DIMENSION; i++) {
      assertSame("Iteration " + i, pattern.getGene(i), ((EvBinaryVectorIndividual)result.get(0)).getGene(i));
    }
  }
}
