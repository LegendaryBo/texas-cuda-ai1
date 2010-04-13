package pl.wroc.uni.ii.evolution.experimental.decisiontree;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvAnswer;
import pl.wroc.uni.ii.evolution.experimental.decisiontree.EvIsEven;

public class EvIsEvenTest extends TestCase {

  public void testDecide() {
    EvIsEven decider = new EvIsEven();
    assertTrue(decider.decide(0) == EvAnswer.YES);
    assertTrue(decider.decide(2) == EvAnswer.YES);
    assertTrue(decider.decide(-2) == EvAnswer.YES);
    assertTrue(decider.decide(6) == EvAnswer.YES);
    assertTrue(decider.decide(-6) == EvAnswer.YES);
    assertTrue(decider.decide(1) == EvAnswer.NO);
    assertTrue(decider.decide(3) == EvAnswer.NO);
    assertTrue(decider.decide(-3) == EvAnswer.NO);
    assertTrue(decider.decide(11) == EvAnswer.NO);
    assertTrue(decider.decide(-15) == EvAnswer.NO);
        
  }

}
