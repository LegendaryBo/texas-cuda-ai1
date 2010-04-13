package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.omega;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvOmegaIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.objectivefunctions.omega.EvOmegaMaxFixedPointsNumber;
import junit.framework.TestCase;

/**
 * @author Rafal Paliwoda (rp@message.pl)
 * @author Mateusz Malinowski (m4linka@gmail.com)
 */
public class EvOmegaLengthReductionOperatorTest extends TestCase {
  
  private final int population_length = 30;
  private final int problem_size = 100;
  private final int chromosome_length = 90;
  private final int lower_bound = 1;
  private final double reduction_factor = 0.5; 
  private final int max_iteration = 10;
  
  public void testOnlyApply() {
    for(int iteration = 0; iteration < max_iteration; iteration++) {
      EvOmegaIndividual template = new EvOmegaIndividual(problem_size, null);
      EvPopulation<EvOmegaIndividual> pop = new EvPopulation<EvOmegaIndividual>();
      for(int i = 0; i < population_length; i++) {
        EvOmegaIndividual ind = new EvOmegaIndividual(
            problem_size, null, chromosome_length);
        ind.setObjectiveFunction(new EvOmegaMaxFixedPointsNumber());
        ind.setTemplate(template);
        pop.add(ind);
      }
      EvOperator<EvOmegaIndividual> length_red_op = 
        new EvOmegaLengthReductionOperator(reduction_factor, lower_bound);
      EvPopulation<EvOmegaIndividual> new_pop = length_red_op.apply(pop);
    
      // we check that all individuals has expected length (= lower_bound)
      boolean isnot_ok = false;
      for(EvOmegaIndividual ind : new_pop) {
        if(ind.getChromosomeLength() != lower_bound) {
          isnot_ok = true;
          break;
        }
      }
      
      if(isnot_ok) {
        fail("All individuals must have lower bound genes");
      }
    }
  }
}
