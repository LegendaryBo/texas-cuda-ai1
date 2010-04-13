package pl.wroc.uni.ii.evolution.experimental.decisiontree;

/**
 * Simple decider if number is even
 */
public class EvIsEven implements EvDecision<Integer> {

  public EvAnswer decide(Integer arg) {
    if (arg % 2 == 0) {
      return EvAnswer.YES;
    } else {
      return EvAnswer.NO;
    }
  }


  @Override
  public int hashCode() {
    return 31;
  }

}
