package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvRouletteSelection;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.fitness.EvIndividualFitness;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.EvBinaryVectorCOMITOperator;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvKDeceptiveOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

// TODO PROPER TEST NEEDED!

public class EvBinaryVectorCOMITOperatorTest  extends TestCase {


  public void testComit() {
    
    int pop_size = 1000;
    int bits = 40;
    
    EvAlgorithm<EvBinaryVectorIndividual> alg = new EvAlgorithm<EvBinaryVectorIndividual>(pop_size);
    EvKDeceptiveOneMax one_max = new EvKDeceptiveOneMax(3);
    EvBinaryVectorSpace solution_space = new EvBinaryVectorSpace(one_max, bits);
    EvBinaryVectorCOMITOperator comit = new EvBinaryVectorCOMITOperator(pop_size);
    //EvTournamentSelection<EvBinaryVectorIndividual> selection = 
    //  new EvTournamentSelection<EvBinaryVectorIndividual>(16,4);
    EvRouletteSelection<EvBinaryVectorIndividual> selection = 
      new EvRouletteSelection<EvBinaryVectorIndividual>(new EvIndividualFitness<EvBinaryVectorIndividual>(),pop_size);
    alg.setTerminationCondition(new EvMaxIteration<EvBinaryVectorIndividual>(30));
    alg.setObjectiveFunction(one_max);
    alg.setSolutionSpace(solution_space);
    alg.addOperatorToEnd(comit);
    alg.addOperatorToEnd(selection);
    long cur_time = System.currentTimeMillis();    
    alg.init();
    
    alg.run();
    
    //assertEquals(alg.getBestResult().toString(), "0000000000000000000000000000000000000000");
    long after_time = System.currentTimeMillis();
    System.out.println("RUN TIME: " + (after_time - cur_time) + "mili sec");
    System.out.print("best: ");           
    
    
  }
  
}
