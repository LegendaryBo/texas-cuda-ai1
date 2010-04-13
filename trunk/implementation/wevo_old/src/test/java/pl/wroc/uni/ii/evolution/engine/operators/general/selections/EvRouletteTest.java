package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRouletteSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvIndividualFitness;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperatorWithBinaryIndividuals;

public class EvRouletteTest extends EvOperatorWithBinaryIndividuals {

  @Override
  protected EvOperator<EvBinaryVectorIndividual> operatorUnderTest() {
    return new EvRouletteSelection<EvBinaryVectorIndividual>(new EvIndividualFitness<EvBinaryVectorIndividual>(),2);
  }

}
