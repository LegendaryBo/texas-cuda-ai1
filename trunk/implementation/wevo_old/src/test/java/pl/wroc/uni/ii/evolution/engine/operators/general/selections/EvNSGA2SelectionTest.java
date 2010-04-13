package pl.wroc.uni.ii.evolution.engine.operators.general.selections;

import static org.junit.Assert.assertTrue;
import org.junit.Test;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvBinaryPattern;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;

/** @author Adam Palka */
public class EvNSGA2SelectionTest {

  /** Test for EvBinaryVectorIndividual. */
  @Test
  public void testEvNSGA2Selection() {
    EvPopulation<EvBinaryVectorIndividual> population =
      new EvPopulation<EvBinaryVectorIndividual>();    
    population.add(new EvBinaryVectorIndividual(new int[] {1, 1, 1, 1}));
    population.add(new EvBinaryVectorIndividual(new int[] {0, 0, 0, 0}));
    for (EvBinaryVectorIndividual b : population) {
      b.addObjectiveFunction(new EvBinaryPattern(new int[] {0, 1, 1, 1}));
      b.addObjectiveFunction(new EvOneMax());
    }
    EvOperator<EvBinaryVectorIndividual> o =
      new EvNSGA2Selection<EvBinaryVectorIndividual>(1);
    population = o.apply(population);
    assertTrue("Wrong population size after selection",
        population.size() == 1);
    assertTrue("Individual was copied witch wrong number"
        + " of objective functions",
        population.get(0).getObjectiveFunctions().size() == 2);
    assertTrue("Wrong individual returned by selection",
        population.get(0).getObjectiveFunctionValue(0) == 3);
    assertTrue("Wrong individual returned by selection",
        population.get(0).getObjectiveFunctionValue(1) == 4);
  }
}
