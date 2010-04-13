package pl.wroc.uni.ii.evolution.distribution;

import org.jmock.Mock;
import org.jmock.MockObjectTestCase;
import pl.wroc.uni.ii.evolution.distribution.clustering.EvSolutionSpaceLoaderFromServlet;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

public class EvSpaceFromServletLoaderTest extends MockObjectTestCase {

  private static final long task_id = 10L;
  private static final long cell_id = 13L;
  private static final int delay = 100;
  
  
  Mock gateway = mock (EvDBServletCommunication.class);
  EvSolutionSpaceLoaderFromServlet loader = 
    new EvSolutionSpaceLoaderFromServlet(task_id, cell_id, (EvDBServletCommunication) gateway.proxy(), delay);
  
  public void testSome() throws Exception {
    
    gateway.expects(once()).method("getVersionOfNewSolutonSpace").will(returnValue(0));
  
    assertFalse(loader.newSubspaceAvailable());
    loader.update();
    assertFalse(loader.newSubspaceAvailable());
    
  }
  
  public void testFetchAvailableSubspace() throws Exception {
    EvBinaryVectorSpace subspace = new EvBinaryVectorSpace(new EvOneMax(), 2);
    
    gateway.expects(once()).method("getVersionOfNewSolutonSpace").will(returnValue(1));
    gateway.expects(once()).method("getSolutionSpace").will(returnValue(subspace));
    
    
    loader.update();
    assertTrue(loader.newSubspaceAvailable());
    assertEquals(subspace, loader.takeSubspace());
    assertFalse(loader.newSubspaceAvailable());
    
    // will not fetch the same subspace again
    
    gateway.expects(once()).method("getVersionOfNewSolutonSpace").will(returnValue(1));
    gateway.expects(never()).method("getSolutionSpace");
    loader.update();
    assertFalse(loader.newSubspaceAvailable());
  }
  
  
}