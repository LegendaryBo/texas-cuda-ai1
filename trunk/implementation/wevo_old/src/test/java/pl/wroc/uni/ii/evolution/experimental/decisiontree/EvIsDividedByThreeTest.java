package pl.wroc.uni.ii.evolution.experimental.decisiontree;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvAnswer;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvIsDividedByThree;

public class EvIsDividedByThreeTest extends TestCase {

  public void testDecideInteger() {
    EvIsDividedByThree decider = new EvIsDividedByThree();
    assertTrue(decider.decide(12) == EvAnswer.YES);
    assertTrue(decider.decide(3) == EvAnswer.YES);
    assertTrue(decider.decide(27) == EvAnswer.YES);
    assertTrue(decider.decide(0) == EvAnswer.YES);
    assertTrue(decider.decide(-12) == EvAnswer.YES);
    assertTrue(decider.decide(-3) == EvAnswer.YES);
    
    assertTrue(decider.decide(2) == EvAnswer.NO);
    assertTrue(decider.decide(-5) == EvAnswer.NO);
    assertTrue(decider.decide(7) == EvAnswer.NO);
    assertTrue(decider.decide(-11) == EvAnswer.NO);
  }

}
