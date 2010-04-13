package pl.wroc.uni.ii.evolution.distribution;

import java.util.Arrays;
import org.jmock.Mock;
import org.jmock.cglib.MockObjectTestCase;
import pl.wroc.uni.ii.evolution.distribution.strategies.EvIslandModel;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchanger;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvSender;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.ExSenderImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiver;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiverImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopology;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvFullConnected;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvIslandDistribution;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;


class ConstantBinaryStrings  extends EvBinaryVectorSpace {

  private static final long serialVersionUID = -5178846553403271939L;

  @Override
  public EvBinaryVectorIndividual generateIndividual() {
    EvBinaryVectorIndividual ind = new EvBinaryVectorIndividual(this.getDimension());
    
    for (int i = 0; i < this.getDimension(); i++) {
      ind.setGene(i, i%2);
    }
    ind.setObjectiveFunction(new EvOneMax());
    return ind;
  }
  public EvBinaryVectorIndividual getAllOnces() {
    EvBinaryVectorIndividual ind = new EvBinaryVectorIndividual(this.getDimension());
    for (int i = 0; i < this.getDimension(); i++) {
      ind.setGene(i, 1);
    }
    return ind;
  }

  public ConstantBinaryStrings(int i, EvObjectiveFunction<EvBinaryVectorIndividual> objective_function ) {
    super(objective_function, i);
  }
}

class Wait implements EvOperator<EvBinaryVectorIndividual> {

  public EvPopulation<EvBinaryVectorIndividual> apply(EvPopulation<EvBinaryVectorIndividual> population) {
    try {
      Thread.sleep(1000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    return population;
  }
  
}


public class EvSpaceLoaderTest extends MockObjectTestCase {

  final long node_id = 123L;
  final long task_id = 55L;
  final long cell_id = 111;
  final int buffer_size = 15;

  
  // Simple test. Population is set of individuals: 01010101011010. ObjectiveFunction is OneMax.
  // Importer has better individuals on queue: 111111111. At end of task, all individuals from population should be
  // 11111111111.
  @SuppressWarnings("unchecked")
  public void testOne() throws InterruptedException {
    EvAlgorithm<EvBinaryVectorIndividual> empty = new EvAlgorithm<EvBinaryVectorIndividual>(30);
    
    EvOneMax objective_function = new EvOneMax();
    
    ConstantBinaryStrings space = new ConstantBinaryStrings(30,objective_function);
    empty.setSolutionSpace(space);
    empty.setObjectiveFunction(objective_function);
    empty.setTerminationCondition(new EvMaxIteration(10));
    empty.addOperatorToEnd(new Wait());
    
    EvIslandDistribution task = new EvIslandDistribution();
    task.setAlgorithm(empty);
    
    
    // distributed topology
    EvTopology top = new EvFullConnected(2);    
    
    Mock mock_exchanger = mock (EvIndividualsExchanger.class);
    
    EvBinaryVectorIndividual ind = space.getAllOnces();
    ind.setObjectiveFunction(new EvOneMax());
    mock_exchanger.expects(atLeastOnce()).method("importIndividuals").with(or (eq(new long[]{0}), eq(new long[] {1})),
         eq(task_id), ANYTHING).will(returnValue(Arrays.asList(new EvBinaryVectorIndividual[] {ind})));

    mock_exchanger.expects(atLeastOnce()).method("exportIndividuals");
    
    
    EvIndividualsExchanger<EvBinaryVectorIndividual> ex = (EvIndividualsExchanger<EvBinaryVectorIndividual>) mock_exchanger.proxy();
    
    EvReceiver imps = new EvReceiverImpl<EvBinaryVectorIndividual>(ex, top, buffer_size, 20, task_id, 100, true);
    EvSender exp  = new ExSenderImpl<EvBinaryVectorIndividual>(ex, top, buffer_size, 20, task_id, node_id, 100,  false);
    
    EvIslandModel<EvBinaryVectorIndividual> strategy = new EvIslandModel<EvBinaryVectorIndividual>(imps,
        exp, new EvBestFromUnionReplacement<EvBinaryVectorIndividual>(), new EvKBestSelection<EvBinaryVectorIndividual>(10));
    
    task.setDistributedStrategy(strategy);
    
    task.run();
       
    assertEquals(30.0, task.getBestResult().getObjectiveFunctionValue());
    for (EvIndividual in : empty.getPopulation()) {
      assertEquals(30.0, in.getObjectiveFunctionValue());
    }
    
  }
  
  
  class WaitStunt implements EvOperator<EvStunt> {

    public EvPopulation<EvStunt> apply(EvPopulation<EvStunt> population) {
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {
      }
      return population;
    }
    
  }
  
}
