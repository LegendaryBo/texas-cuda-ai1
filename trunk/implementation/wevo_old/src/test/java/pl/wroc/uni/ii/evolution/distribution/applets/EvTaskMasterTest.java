package pl.wroc.uni.ii.evolution.distribution.applets;

import org.jmock.Mock;
import org.jmock.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.distribution.workers.EvEvolutionInterface;
import pl.wroc.uni.ii.evolution.distribution.workers.EvJARCache;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskLoader;
import pl.wroc.uni.ii.evolution.distribution.workers.EvTaskMaster;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;

public class EvTaskMasterTest extends MockObjectTestCase {
 
	static class TaskMock implements Runnable {
	    public long counter = 0;

      
	    public void run() {
	      while(true) {
	        counter++;
	        try {
	          Thread.sleep(10);
	        } catch (InterruptedException e) {
	          return;
	        }
	      }
	    }
      
	    public boolean stopped() {
	      long counter_value = counter;
	      try {
	        Thread.sleep(20);
	      } catch (InterruptedException e) {
	        e.printStackTrace();
	      }
	      return counter_value == counter;
	    }
	  }

  Mock servlet_proxy = mock(EvManagmentServletCommunication.class);
  Mock loader = mock(EvTaskLoader.class);
  Mock jar_cache = mock(EvJARCache.class);
  EvTaskMaster island_master;
  
  @Override
  protected void setUp() throws Exception {
    
    island_master = new EvTaskMaster((EvManagmentServletCommunication) servlet_proxy.proxy(), (EvTaskLoader)loader.proxy(),
        (EvJARCache) jar_cache.proxy(), 200,0,(EvEvolutionInterface)null);
        
    servlet_proxy.stubs().method("getNodeID").will(returnValue(2L));
    servlet_proxy.stubs().method("getURL").will(returnValue("http://localhost:8081/wevo_system/ManagementServlet"));
    servlet_proxy.stubs().method("getTaskID").will(returnValue(4));
    servlet_proxy.stubs().method("keepAlive").will(returnValue(true));
    jar_cache.stubs().method("getJARUrl").will(returnValue("home.dir/wevo_tasks/3/task.jar"));
  }
  
  @Override
  protected void tearDown() throws Exception {
    island_master.stop();
  }

  public void testReportProblemWithGetClientID() throws InterruptedException {
    servlet_proxy.stubs().method("getNodeID").will(
        throwException(new Exception()));
    island_master.start();
    Thread.sleep(400);
  }

  public void testReportProblemWithGetTaskID() throws Exception {
    servlet_proxy.expects(atLeastOnce()).method("getTaskID").will(
        throwException(new Exception()));
    island_master.start();
    Thread.sleep(400);
    island_master.stop();
  }


  public void testStopsTheTaskWhenKeepAliveReturnsFalse() throws Exception {
	  //actually will return false at third call, because there is
	    //already one call sqeduled by the setup
	    servlet_proxy.expects(atLeastOnce()).method("keepAlive").with(ANYTHING).will(returnValue(false));
	    TaskMock task = new TaskMock();
	    loader.expects(atLeastOnce()).method("getTask").will(returnValue(task));
	    //loader.stubs().method("getTask").will(onConsecutiveCalls(returnValue(task), returnValue(new TaskMock())));
	    island_master.start();
	    Thread.sleep(3000);
	    island_master.stop();
	    Thread.sleep(200);
	    
	    assertTrue(task.stopped());
  }
}