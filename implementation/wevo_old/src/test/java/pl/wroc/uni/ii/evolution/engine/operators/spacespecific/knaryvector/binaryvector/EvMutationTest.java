package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorNegationMutation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperatorWithBinaryIndividuals;

public class EvMutationTest extends EvOperatorWithBinaryIndividuals {

  @Override
  protected EvOperator<EvBinaryVectorIndividual> operatorUnderTest() {
    return new EvBinaryVectorNegationMutation(0.2);
  }

}
