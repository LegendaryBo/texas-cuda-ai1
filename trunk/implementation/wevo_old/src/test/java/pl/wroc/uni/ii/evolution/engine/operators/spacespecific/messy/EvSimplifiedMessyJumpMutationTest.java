package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvSimplifiedMessyIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvSimplifiedMessyJumpMutation;



public class EvSimplifiedMessyJumpMutationTest extends TestCase {

  public void testOnlyApply() {
    EvSimplifiedMessyIndividual individual = new EvSimplifiedMessyIndividual(1);
    EvSimplifiedMessyJumpMutation mutation = new EvSimplifiedMessyJumpMutation(1.0);
    mutation.mutate(individual);
    
    
  }

}
