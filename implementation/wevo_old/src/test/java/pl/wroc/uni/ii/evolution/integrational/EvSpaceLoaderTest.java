package pl.wroc.uni.ii.evolution.integrational;

import pl.wroc.uni.ii.evolution.distribution.clustering.EvSolutionSpaceLoaderFromServlet;
import pl.wroc.uni.ii.evolution.distribution.tasks.EvIslandDistribution;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskThread;
import pl.wroc.uni.ii.evolution.engine.EvAlgorithm;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.engine.terminationconditions.EvMaxIteration;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.testhelper.EvGoalFunction;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;
import pl.wroc.uni.ii.evolution.testhelper.EvStuntSpace;
import junit.framework.TestCase;

class WaitStunt implements EvOperator<EvStunt> {

  public EvPopulation<EvStunt> apply(EvPopulation<EvStunt> population) {
    try {
      Thread.sleep(1000);
    } catch (InterruptedException e) {
    }
    return population;
  }
  
}





public class EvSpaceLoaderTest extends TestCase {
  
  
  final long node_id = 123L;
  final long task_id = 55L;
  final long cell_id = 111;
  final int buffer_size = 15;
  
  
  //at begging of algorithm population is set of following ind:
  //1, 3, 5, 7, 9, 11, 13, ...
  // after some delay ... spaceloader discovered new subspace and loads it
  // then population become 0, 2, 4, 6, 8, ...
  
  public void testSpaceLoaderWorks() throws Exception {
    

//    String wevo_server_url = "http://127.0.0.1:8080";
//
//    
//    // empty alg
//    EvAlgorithm<EvStunt> empty = new EvAlgorithm<EvStunt>(30);
//    
//    empty.setSolutionSpace(new EvStuntSpace(1));
//    empty.setObjectiveFunction(new EvGoalFunction());
//    empty.setTerminationCondition(new EvMaxIteration<EvStunt>(10));
//    empty.addOperatorToEnd(new WaitStunt());
//    
//    EvIslandDistribution task = new EvIslandDistribution();
//    task.setAlgorithm(empty);
//    
//    EvDBServletCommunicationImpl gateway = new EvDBServletCommunicationImpl(wevo_server_url);
//    
//    gateway.deleteSolutionSpaces(task_id);
//    
//    EvSolutionSpaceLoaderFromServlet loader = new EvSolutionSpaceLoaderFromServlet(task_id, cell_id, gateway, 1000); 
//    task.setSolutionSpaceLoader(loader);
//    
//    EvTaskThread thread = new EvTaskThread(task);
//   
//    System.out.println("Start");
//    thread.start();
//    Thread.sleep(2000);
//    gateway.addSolutionSpace(task_id, cell_id, new EvStuntSpace(0));
//    Thread.sleep(3000);
//    thread.stop();
//    System.out.println("Stop");
//    
//    for (EvStunt ind: empty.getPopulation()) {
//      assertTrue(ind.getObjectiveFunctionValue() % 2 == 0);
//    }
    
  }  
}
