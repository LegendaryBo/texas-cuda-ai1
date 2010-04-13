package pl.wroc.uni.ii.evolution.experimental.decisiontree;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvAnswer;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvIsDividedByFive;

public class EvIsDividedByFiveTest extends TestCase {

  public void testDecide() {
    EvIsDividedByFive decider = new EvIsDividedByFive();
    assertTrue(decider.decide(15) == EvAnswer.YES);
    assertTrue(decider.decide(5) == EvAnswer.YES);
    assertTrue(decider.decide(20) == EvAnswer.YES);
    assertTrue(decider.decide(0) == EvAnswer.YES);
    assertTrue(decider.decide(-35) == EvAnswer.YES);
    assertTrue(decider.decide(-5) == EvAnswer.YES);
    
    assertTrue(decider.decide(1) == EvAnswer.NO);
    assertTrue(decider.decide(-1) == EvAnswer.NO);
    assertTrue(decider.decide(3) == EvAnswer.NO);
    assertTrue(decider.decide(-3) == EvAnswer.NO);
    
  }

}
