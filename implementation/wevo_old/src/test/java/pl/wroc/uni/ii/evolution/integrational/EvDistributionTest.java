package pl.wroc.uni.ii.evolution.integrational;

import java.io.IOException;
import org.jmock.Mock;
import org.jmock.cglib.MockObjectTestCase;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.EvIslandModel;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchangeWithServlet;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchanger;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiverImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.ExSenderImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopology;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvIslandDistribution;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskThread;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvBestFromUnionReplacement;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.prototype.EvTask;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.*;
import pl.wroc.uni.ii.evolution.testhelper.EvGoalFunction;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;
import pl.wroc.uni.ii.evolution.testhelper.EvStuntSpace;

/**
 * Important test that shows that distribution works !!!
 * It uses external servlets.  
 * @author Marcin Golebiowski
 */
public class EvDistributionTest extends MockObjectTestCase {
  
	public void testDummy() {
		
	}
	
//  private String wevo_server_url = "http://127.0.0.1:8080";
//  
//  
//  class Node<T extends EvIndividual> {
//    public EvTaskThread thread;
//    public EvTask task;
//    public EvAlgorithm<T> alg;
//  }
//  
//  
//  // simple operator that delays execution of doIteration
//  class Wait implements EvOperator<EvStunt> {
//
//    private int delay;
//    
//    public Wait(int delay) {
//      this.delay = delay; 
//    }
//    
//    public EvPopulation<EvStunt> apply(EvPopulation<EvStunt> population) {
//      
//      try {
//        Thread.sleep(delay);
//      } catch (InterruptedException e) {}
//      
//      
//      return population;
//    }
//    
//  }
//  
//  
//  // add some individual to population once after some delay
//  class AddSomeIndividualAfterSomeDelay implements EvOperator<EvStunt> {
//
//    private int sleep;
//    private long start;
//    private boolean added = false;
//    private EvStunt t;
//    
//    public AddSomeIndividualAfterSomeDelay(int sleep, EvStunt t) {
//      this.sleep = sleep;
//      this.start = System.currentTimeMillis();
//      this.t = t;
//    }
//    
//    public EvPopulation<EvStunt> apply(EvPopulation<EvStunt> population) {
//      if (!added && (System.currentTimeMillis() - start > sleep)) {
//        added = true;
//        population.add(t);
//      }
//      
//      return population;
//    }
//  }
//  
//  
//  // add some individual to population if 
//  class AddSomeIndividualIfSomeIndividualExist implements EvOperator<EvStunt> {
//
//    private EvStunt some;
//    private EvStunt to_add;
//    private boolean added = false;
//    
//    public AddSomeIndividualIfSomeIndividualExist(EvStunt some, EvStunt to_add) {
//      this.some = some;
//      this.to_add = to_add;
//    }
//    
//    
//    public EvPopulation<EvStunt> apply(EvPopulation<EvStunt> population) {
//      if (population.contains(some) && !added) {
//        population.add(to_add);
//        added = true;
//      }
//      return population;
//    }
//  }
//  
//
//  
//  
//
//  public Node<EvStunt> createCustomTaskForNode(int iter, long task_id, long node_id, long cell_id, long[] neigh, int pop_size,
//      int parity, int buffer_size, int count, boolean duplicates, int export_count) {
//    
//    EvAlgorithm<EvStunt> alg = new EvAlgorithm<EvStunt>(pop_size);
//    alg.setTerminationCondition(new EvMaxIteration<EvStunt>(iter));
//    alg.setObjectiveFunction(new EvGoalFunction());
//    alg.setSolutionSpace(new EvStuntSpace(parity));
//    alg.addOperatorToEnd(new Wait(4000));    
//    
//    EvIslandDistribution task = new EvIslandDistribution();
//    task.setAlgorithm(alg);
//   
//    Mock top = mock (EvTopology.class);
//    
//    top.stubs().method("assignCellID").will(returnValue(cell_id));
//    top.stubs().method("getNeighbours").will(returnValue(neigh));
//    
//
//    EvTopology topology = (EvTopology) top.proxy();
//    EvIndividualsExchanger<EvStunt> exchanger = new EvIndividualsExchangeWithServlet<EvStunt>(new EvDBServletCommunicationImpl(wevo_server_url));
//    
//    EvIslandModel strategy = new EvIslandModel<EvStunt>(
//        new EvReceiverImpl<EvStunt>(exchanger, topology, buffer_size, 100, task_id, count, duplicates),
//        new ExSenderImpl<EvStunt>(exchanger, topology, buffer_size, 100, task_id, node_id, count,  duplicates),
//        new EvBestFromUnionReplacement<EvStunt>(),
//        new EvKBestSelection<EvStunt>(export_count));
//    
//    task.setDistributedStrategy(strategy);
//    EvTaskThread thread = new EvTaskThread(task);
//    Node<EvStunt> s = new Node<EvStunt>();
//    s.alg = alg;
//    s.task = task;
//    s.thread = thread;
//
//   return s;
//  }
//  
//  
//  // at end of this experiment two nodes must have population consisting of all 9
//  public void testTwoNodesCooperating() throws InterruptedException, IOException {
//  
//    
//    EvDBServletCommunicationImpl db = new EvDBServletCommunicationImpl(wevo_server_url);
//    
//    Node node1, node2;
//    long task_id = 667;
//    
//    db.deleteIndividualsFromTask(task_id);
//    
//    node1 = createCustomTaskForNode(5, task_id, 111, 0, new long[] { 1 }, 5, 0, 10, 1, true, 1);
//    node2 = createCustomTaskForNode(5, task_id, 112, 1, new long[] { 0 }, 5, 1, 10, 1, true, 1);
//    
//   
//    node1.thread.start();
//    node2.thread.start();
//    
//    Thread.sleep(10000);
//    
//    node1.thread.stop();
//    node2.thread.stop();
//    
//    
//    for (Object ind: node1.alg.getPopulation()) {
//      assertEquals(9.0, ((EvIndividual) ind).getObjectiveFunctionValue());
//    }
//    
//   
//    for (Object ind: node2.alg.getPopulation()) {
//      assertEquals(9.0, ((EvIndividual) ind).getObjectiveFunctionValue());
//    }
//    
//    db.deleteIndividualsFromTask(task_id);
//   
//  }
//  
//  
//  
//  // 110 created in node1
//  // then in node2 111 is discovered because of 110  received from node1
//  // then in node3 112 is discovered because of 111  received from node2
//  // then 112 is sended to node1
//  
//  //  <<0>> --> <<1>> --> <<2>> --> <<0>>
//  
//
//  public void testThreeNodesCooperating() throws Exception {
//    long task_id = 667;
//    EvDBServletCommunicationImpl db = new EvDBServletCommunicationImpl(wevo_server_url);
//    db.deleteIndividualsFromTask(task_id);
//   
//    
//    assertEquals("Clear individual for task failed", 0, db.getIndividualCount(task_id));
//    
//    //definition of node1
//    Node<EvStunt> node1 = createCustomTaskForNode(10, task_id, 10, 0, new long[] {2}, 10, 1, 10, 10, false, 1);
//    EvStunt t2 = new EvStunt(110.0);
//
//    t2.setObjectiveFunction(new EvGoalFunction());
//    
//    node1.alg.addOperatorToEnd(new AddSomeIndividualAfterSomeDelay(100, t2));
//    
//    
//    //definition of node2
//    Node<EvStunt> node2 = createCustomTaskForNode(10, task_id, 11, 1, new long[] {0}, 10, 1, 10, 10, false, 1);
//    EvStunt t3 = new EvStunt(111.0);
//    t3.setObjectiveFunction(new EvGoalFunction());
//    
//    node2.alg.addOperatorToEnd(new AddSomeIndividualIfSomeIndividualExist(t2, t3));
//    
//    
//    //definition of node3
//    Node<EvStunt> node3 = createCustomTaskForNode(10, task_id, 12, 2, new long[] {1}, 10, 1, 10, 10, false, 1);
//    EvStunt t4 = new EvStunt(112.0);
//    t4.setObjectiveFunction(new EvGoalFunction());
//
//    node3.alg.addOperatorToEnd(new AddSomeIndividualIfSomeIndividualExist(t3, t4));
//    
//
//    node1.thread.start();
//    node2.thread.start();
//    node3.thread.start();
//    
//    
//    Thread.sleep(10000);
//    node1.thread.stop();
//    node2.thread.stop();
//    node3.thread.stop();
//    
//   
//    System.out.println("Node1");
//    for (Object o: node1.alg.getPopulation()) {
//      System.out.println(o);
//    }
//    
//    System.out.println("Node2");
//    for (Object o: node2.alg.getPopulation()) {
//      System.out.println(o);
//    }
//    
//    System.out.println("Node3");
//    for (Object o: node2.alg.getPopulation()) {
//      System.out.println(o);
//    }
//    
//  
//    assertTrue("Population in node1 must have t4", node1.alg.getPopulation().contains(t4));    
//    assertTrue("Population in node2 must have t4", node2.alg.getPopulation().contains(t4));
//    assertTrue("Population in node3 must have t4", node3.alg.getPopulation().contains(t4));
//
//    db.deleteIndividualsFromTask(task_id);
//    
//  
//  }
//  
//  
//  public void testShowsThatExportingIndividualsWorks() throws Exception {
//
//    int nr_iter = 11;
//    int select = 10;
//    
//   
//    Node node1 = createCustomTaskForNode(nr_iter, 666, 1, 1, new long[] {2}, 100, 1, 10, 10 , false, select);
//   
//    EvDBServletCommunicationImpl db = new EvDBServletCommunicationImpl(wevo_server_url);
//    db.deleteIndividualsFromTask(666);
//    
//    node1.thread.start();
//    Thread.sleep(20000);
//    node1.thread.stop();
//    Thread.sleep(1000);
//    assertEquals((nr_iter - 1) * 10, db.getIndividualCount(666));
//    
//    db.deleteIndividualsFromTask(666);
//  }

 
}
