package pl.wroc.uni.ii.evolution.experimental.decisiontree;

/**
 * Simple decider if number is divided by 5
 */
public class EvIsDividedByFive implements EvDecision<Integer> {

  public EvAnswer decide(Integer arg) {
    if (arg % 5 == 0) {
      return EvAnswer.YES;
    } else {
      return EvAnswer.NO;
    }

  }


  @Override
  public int hashCode() {
    return 13;
  }

}
